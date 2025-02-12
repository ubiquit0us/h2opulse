import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from rembg import remove

def edge_detection(input_img):
    """Applies Gaussian blur, converts to grayscale, thresholds, and detects edges."""
    blur = cv2.GaussianBlur(input_img, (15, 15), 0)
    gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    edges = cv2.Canny(thresh, 50, 100, apertureSize=3)
    return cv2.erode(cv2.dilate(edges, None), None)

def hand_gesture(img):
    """Detects hand gestures and finds key points like valleys and convex hull."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.bitwise_not(thresh)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img, [], [], [], (0, 0)
    
    max_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(max_contour)
    moments = cv2.moments(max_contour)
    
    if moments['m00'] == 0:
        return img, [], [], [], (0, 0)
    
    center = (int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00']))
    
    defects = cv2.convexityDefects(max_contour, cv2.convexHull(max_contour, returnPoints=False)) if len(max_contour) >= 5 else None
    valley_points, start_points, end_points = [], [], []
    
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, _ = defects[i, 0]
            valley_points.append(tuple(max_contour[f][0]))
            start_points.append(tuple(max_contour[s][0]))
            end_points.append(tuple(max_contour[e][0]))
    
    return img, valley_points, start_points, end_points, center

def center_calc(input_img):
    """Calculates and refines the center of the hand gesture."""
    img, valley_points, start_points, end_points, center = hand_gesture(input_img)
    rotated_img = input_img.copy()
    
    while True:
        img_cropped = rotated_img[:center[1], :]
        _, _, _, _, new_center = hand_gesture(img_cropped)
        
        center1_y = center[1] - (center[1] - new_center[1]) // 3
        center1_x = int((center1_y - center[1]) * (new_center[0] - center[0]) / (new_center[1] - center[1]) + center[0])
        center1 = (center1_x, center1_y)
        
        angle = np.arctan2(center1[0] - center[0], center1[1] - center[1]) * 180 / np.pi
        M = cv2.getRotationMatrix2D(center1, -angle, 1)
        rotated_img = cv2.warpAffine(rotated_img, M, (input_img.shape[1], input_img.shape[0]))
        
        edges = edge_detection(rotated_img)
        topmost = next(((center1[0], y) for y in range(center1[1], 0, -1) if edges[y, center1[0]] == 255), None)
        if topmost is None:
            break
        
        dist_topmost = np.linalg.norm(np.array(topmost) - np.array(center1))
        if dist_topmost < 30:
            break
        center = center1
    
    return rotated_img, center1, angle

def get_roi(input_img_loc):
    """Extracts the Region of Interest (ROI) from the hand image."""
    input_img = cv2.imread(input_img_loc)
    input_img[:2, :] = input_img[:, :2] = 0
    input_img = cv2.cvtColor(remove(input_img, bgcolor=[0, 0, 0, 255]), cv2.COLOR_BGR2RGB)
    
    img, center, angle = center_calc(input_img)
    M = cv2.getRotationMatrix2D(center, -angle, 1)
    img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    
    edges = edge_detection(img)
    all_points = np.column_stack(np.where(edges == 255))
    
    min_dist, min_index = min((np.linalg.norm(pt - center), i) for i, pt in enumerate(all_points))
    
    cropped_img = img[center[1]-int(min_dist):center[1]+int(min_dist), center[0]-int(min_dist):center[0]+int(min_dist)]
    cv2.imwrite('cropped_img.jpg', cropped_img)
    
    return input_img, edges, img, cropped_img

def plot_roi_images_steps(original_img, edges, img, cropped_img):
    """Plots various steps of ROI extraction."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    images = [original_img, edges, img, cropped_img]
    titles = ['Original Image', 'Edge Image', 'Processed Hand Image', 'ROI Circle']
    
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if len(img.shape) == 3 else img, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    
    plt.show()

def process_images(input_img_locs):
    """Processes all images in a given directory."""
    for img_file in os.listdir(input_img_locs):
        if img_file.endswith('.jpg'):
            input_img_loc = os.path.join(input_img_locs, img_file)
            print(f'Processing: {input_img_loc}')
            original_img, edges, img, cropped_img = get_roi(input_img_loc)
            plot_roi_images_steps(original_img, edges, img, cropped_img)

# Example usage
input_img_locs = ''
process_images(input_img_locs)
