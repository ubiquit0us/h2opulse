DATA_DIR = '/content/gdrive/My Drive/dataset/data/'

import os
import shutil
import cv2
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')
model.train(data='/content/gdrive/My Drive/dataset/config.yaml', epochs=30, imgsz=640)

shutil.make_archive('/content/runs', 'zip', '/content/runs')

model_path = '/content/runs/segment/train/weights/last.pt'
image_path = '/content/gdrive/My Drive/dataset/data/images/'

img = cv2.imread(image_path)
H, W, _ = img.shape

model = YOLO(model_path)

results = model(img)

for result in results:
    for j, mask in enumerate(result.masks.data):
        mask = mask.cpu().numpy() * 255
        mask = cv2.resize(mask, (W, H))

        cv2.imwrite('./output.png', mask)
        plt.imshow(mask)
        plt.show()
