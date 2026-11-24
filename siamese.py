import os, time, argparse
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision
from torchvision.io import decode_image, ImageReadMode
import torchvision.transforms.v2 as T
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score

p = argparse.ArgumentParser()
p.add_argument("--backbone", default="small", choices=["small", "resnet"])
p.add_argument("--epochs", type=int, default=50)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--bs", type=int, default=16)
p.add_argument("--size", type=int, default=200)
p.add_argument("--emb", type=int, default=128)
p.add_argument("--steps_per_epoch", type=int, default=60)
p.add_argument("--seeds", default="42")
p.add_argument("--head", default="distance", choices=["distance", "learned"],
               help="distance: contrastive loss only, classify by cosine distance with a threshold fit on TRAINING subjects")
p.add_argument("--tag", default="")
args = p.parse_args()

DATA_ROOT = os.path.join(os.getcwd(), "Dataset")
OUT = os.path.join(os.getcwd(), "outputs", "with_reference")
os.makedirs(OUT, exist_ok=True)
VIEWS = ["left_hand/dorsal", "right_hand/dorsal", "left_hand/wrist", "right_hand/wrist"]
dev = torch.device("cuda")
torch.backends.cudnn.benchmark = True


def imgs(folder):
    if not os.path.isdir(folder):
        return []
    return sorted(os.path.join(folder, n) for n in os.listdir(folder)
                  if not n.startswith(".") and n.lower().endswith((".jpg", ".jpeg", ".png")))


users = sorted(d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d)) and not d.startswith("."))
paths, index = [], {}
for u in users:
    for st in ["hydrated", "dehydrated"]:
        for v in VIEWS:
            ids = []
            for q in imgs(os.path.join(DATA_ROOT, u, st, *v.split("/"), "yolo_vein_roi")):
                ids.append(len(paths)); paths.append(q)
            index[(u, st, v)] = ids
ok = [u for u in users if all(index[(u, "hydrated", v)] and index[(u, "dehydrated", v)] for v in VIEWS)]
PRE = int(args.size * 1.12)
cache = os.path.join(OUT, f"images_{PRE}.pt")
if os.path.exists(cache):
    X = torch.load(cache)
else:
    X = torch.stack([T.functional.resize(decode_image(q, mode=ImageReadMode.RGB), [PRE, PRE], antialias=True) for q in paths])
    torch.save(X, cache)
X = X.to(dev)
print(f"images {len(paths)}  subjects {len(users)} (with all 4 views in both states: {len(ok)})  backbone {args.backbone}", flush=True)

aug = T.Compose([T.RandomResizedCrop(args.size, scale=(0.75, 1.0), ratio=(0.9, 1.1), antialias=True),
                 T.RandomRotation(8), T.ColorJitter(0.1, 0.1, 0.05)])
MEAN = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)


def prep(ids, train):
    xb = X[torch.as_tensor(ids, device=dev)].float() / 255
    if train:
        n, _, H, W = xb.shape
        s = torch.empty(n, device=dev).uniform_(0.75, 1.0).sqrt()
        w, h = s * W, s * H * torch.empty(n, device=dev).uniform_(0.95, 1.05)
        h = h.clamp(max=H)
        x0 = torch.rand(n, device=dev) * (W - w); y0 = torch.rand(n, device=dev) * (H - h)
        boxes = torch.stack([torch.arange(n, device=dev, dtype=torch.float), x0, y0, x0 + w, y0 + h], 1)
        xb = torchvision.ops.roi_align(xb, boxes, (args.size, args.size), aligned=True)
        b = torch.empty(n, 1, 1, 1, device=dev).uniform_(0.9, 1.1)
        c = torch.empty(n, 1, 1, 1, device=dev).uniform_(0.9, 1.1)
        m = xb.mean((1, 2, 3), keepdim=True)
        xb = ((xb - m) * c + m * b).clamp(0, 1)
    else:
        xb = T.functional.resize(xb, [args.size, args.size], antialias=True)
    return (xb - MEAN) / STD


class SmallCNN(nn.Module):
    def __init__(self, emb):
        super().__init__()
        ch = [3, 32, 64, 128, 128]
        L = []
        for a, b in zip(ch[:-1], ch[1:]):
            L += [nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True), nn.AvgPool2d(2)]
        self.f = nn.Sequential(*L)
        self.fc = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(128, emb))

    def forward(self, x):
        return self.fc(F.adaptive_avg_pool2d(self.f(x), 1))


class Tower(nn.Module):
    def __init__(self):
        super().__init__()
        if args.backbone == "small":
            self.sub = nn.ModuleList([SmallCNN(args.emb) for _ in VIEWS])
            self.shared = None
        else:
            m = torchvision.models.resnet50(weights="IMAGENET1K_V2")
            self.stem = nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1, m.layer2)
            for q in self.stem.parameters():
                q.requires_grad = False
            self.shared = m.layer3
            self.sub = nn.ModuleList([nn.Sequential(nn.Dropout(0.3), nn.Linear(1024, args.emb)) for _ in VIEWS])

    def forward(self, xs):
        out = []
        for i, x in enumerate(xs):
            if self.shared is None:
                out.append(self.sub[i](x))
            else:
                with torch.no_grad():
                    self.stem.eval(); h = self.stem(x)
                h = F.adaptive_avg_pool2d(self.shared(h), 1).flatten(1)
                out.append(self.sub[i](h))
        return torch.cat([F.normalize(o.float(), dim=1) for o in out], 1)


class Siamese(nn.Module):
    def __init__(self):
        super().__init__()
        self.tower = Tower()
        self.bn = nn.BatchNorm1d(1)
        self.out = nn.Linear(1, 1)

    def forward(self, ref, qry):
        n = ref[0].shape[0]
        e = self.tower([torch.cat([r, q]) for r, q in zip(ref, qry)])
        a, b = e[:n], e[n:]
        dist = 1 - F.cosine_similarity(a, b, dim=1, eps=1e-6)
        sim = torch.sigmoid(self.out(self.bn(dist.unsqueeze(1)))).squeeze(1)
        return sim, dist


def sample_batch(subjects, rng, n):
    refs, qrys, lab = [[] for _ in VIEWS], [[] for _ in VIEWS], []
    for _ in range(n):
        u = subjects[rng.integers(len(subjects))]
        st = "hydrated" if rng.random() < 0.5 else "dehydrated"
        for i, v in enumerate(VIEWS):
            H = index[(u, "hydrated", v)]
            r = H[rng.integers(len(H))]
            pool = index[(u, st, v)]
            if st == "hydrated" and len(H) > 1:
                pool = [h for h in H if h != r]
            refs[i].append(r); qrys[i].append(pool[rng.integers(len(pool))])
        lab.append(1.0 if st == "hydrated" else 0.0)
    return refs, qrys, np.array(lab, dtype=np.float32)


def test_samples(subjects, protocol, rng):
    S = []
    for u in subjects:
        ref = [index[(u, "hydrated", v)][rng.integers(len(index[(u, 'hydrated', v)]))] for v in VIEWS]
        for st in ["hydrated", "dehydrated"]:
            per_view = []
            for i, v in enumerate(VIEWS):
                ids = index[(u, st, v)]
                if protocol == "strict" and st == "hydrated" and len(ids) > 1:
                    ids = [h for h in ids if h != ref[i]]
                per_view.append(ids)
            K = max(len(x) for x in per_view)
            for k in range(K):
                S.append((u, st, ref, [ids[k % len(ids)] for ids in per_view]))
    return S


@torch.no_grad()
def predict(model, S):
    model.eval()
    sims = []
    for i in range(0, len(S), 64):
        chunk = S[i:i + 64]
        ref = [prep([s[2][v] for s in chunk], False) for v in range(4)]
        qry = [prep([s[3][v] for s in chunk], False) for v in range(4)]
        sims.append(model(ref, qry)[1].float().cpu())
    return torch.cat(sims).numpy()


def metrics(yt, p_dehyd):
    yh = (p_dehyd >= 0.5).astype(int)
    return dict(acc=accuracy_score(yt, yh), prec=precision_score(yt, yh, zero_division=0),
                rec=recall_score(yt, yh, zero_division=0), f1=f1_score(yt, yh, zero_division=0), auc=roc_auc_score(yt, p_dehyd))


tag = args.tag or f"siamese_{args.backbone}_e{args.epochs}_lr{args.lr}"
summary = []
t0 = time.time()
for seed in [int(s) for s in args.seeds.split(",")]:
    lab_img = np.array([0 if "/hydrated/" in q else 1 for q in paths])
    grp = np.array([os.path.relpath(q, DATA_ROOT).split(os.sep)[0] for q in paths])
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(paths)), lab_img, grp))
    res = {"paper": [], "strict": []}
    for k, (tr, te) in enumerate(folds, 1):
        torch.manual_seed(seed * 10 + k)
        rng = np.random.default_rng(seed * 10 + k)
        s_tr = [u for u in ok if u in set(grp[tr])]
        s_te = [u for u in ok if u in set(grp[te])]
        model = Siamese().to(dev)
        opt = torch.optim.Adam([q for q in model.parameters() if q.requires_grad], lr=args.lr,
                               weight_decay=1e-4 if args.backbone == "small" else 1e-5)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs * args.steps_per_epoch)
        for ep in range(args.epochs):
            model.train()
            for _ in range(args.steps_per_epoch):
                r, q, lab = sample_batch(s_tr, rng, args.bs)
                yb = torch.tensor(lab, device=dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    sim, dist = model([prep(x, True) for x in r], [prep(x, True) for x in q])
                sim, dist = sim.float(), dist.float()
                contrastive = (yb * dist ** 2 + (1 - yb) * F.relu(1.0 - dist) ** 2).mean()
                loss = contrastive + (F.binary_cross_entropy(sim.clamp(1e-6, 1 - 1e-6), yb) if args.head == "learned" else 0)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
        Str = test_samples(s_tr, "strict", np.random.default_rng(seed * 1000 + k + 500))
        dtr = predict(model, Str); ytr = np.array([int(s[1] == "dehydrated") for s in Str])
        cands = np.unique(dtr)
        thr = cands[np.argmax([np.mean((dtr >= c) == ytr) for c in cands])]
        for protocol in res:
            S = test_samples(s_te, protocol, np.random.default_rng(seed * 1000 + k))
            dist = predict(model, S)
            res[protocol].append(pd.DataFrame({"subj": [s[0] for s in S], "y": [int(s[1] == "dehydrated") for s in S],
                                               "p": 1 / (1 + np.exp(-(dist - thr) * 20)), "dist": dist, "fold": k}))
        m = metrics(res["paper"][-1].y, res["paper"][-1].p.values)
        print(f"[{tag}] split {seed} fold {k}: paper-protocol multi-view acc {m['acc']:.3f} F1 {m['f1']:.3f} "
              f"AUC {m['auc']:.3f}  ({time.time()-t0:.0f}s)", flush=True)
    for protocol, parts in res.items():
        d = pd.concat(parts)
        d.to_csv(os.path.join(OUT, f"pred_{tag}_{protocol}_s{seed}.csv"), index=False)
        mv = metrics(d.y.values, d.p.values)
        se = d.groupby(["subj", "y"]).agg(y=("y", "first"), p=("p", "mean"))
        ss = metrics(se.y.values, se.p.values)
        row = {"tag": tag, "split_seed": seed, "protocol": protocol,
               **{f"multiview_{a}": b for a, b in mv.items()}, **{f"session_{a}": b for a, b in ss.items()}}
        summary.append(row)
        print(f"[{tag}] split {seed} {protocol:6s} POOLED: multi-view acc {mv['acc']:.3f} prec {mv['prec']:.3f} "
              f"rec {mv['rec']:.3f} F1 {mv['f1']:.3f} AUC {mv['auc']:.3f} | session acc {ss['acc']:.3f} AUC {ss['auc']:.3f}",
              flush=True)
f = os.path.join(OUT, "siamese_results.csv")
pd.DataFrame(summary).assign(**{k: str(v) for k, v in vars(args).items() if k not in ("seeds", "tag")}).to_csv(
    f, mode="a", header=not os.path.exists(f), index=False)
print(f"[{tag}] done in {time.time()-t0:.0f}s", flush=True)
