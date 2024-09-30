import cv2
import torch
import os
import numpy as np
from matplotlib import pyplot as plt

def load_custom_model(weights_path='best.pt'):
    repo_dir = 'yolov9'
    model = torch.hub.load(repo_dir, 'custom', path=weights_path, source='local', force_reload=True)
    return model

def draw_grid(image, bbox, rows, cols):
    x1, y1, x2, y2 = bbox
    dx, dy = (x2 - x1) // cols, (y2 - y1) // rows
    
    for i in range(1, cols):
        cv2.line(image, (x1 + i * dx, y1), (x1 + i * dx, y2), (0, 0, 255), 2)
        
    for i in range(1, rows):
        cv2.line(image, (x1, y1 + i * dy), (x2, y1 + i * dy), (0, 0, 255), 2)
        
    return image

def detect_and_draw_grid(image_path, rows=4, cols=5, output_path='output_image.jpg'):
    print("Loading custom model...")
    model = load_custom_model()
    
    print(f"Reading image from {image_path}...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image {image_path}")
        return
    
    print("Running detection...")
    results = model(image)
    detections = results.xyxy[0].cpu().numpy()
    labels = results.names
    pool_bbox = None
    max_area = 0

    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        area = (x2 - x1) * (y2 - y1)
        label = labels[int(cls)]
        
        if label == 'swimming-pool' and area > max_area:
            max_area = area
            pool_bbox = (x1, y1, x2, y2)
            confidence = conf * 100

    if pool_bbox:
        print("Swimming pool detected, drawing grid and label...")
        x1, y1, x2, y2 = pool_bbox
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_text = f"swimming-pool {confidence:.0f}%"
        (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(image, (x1, y1 - h - 5), (x1 + w, y1), (0, 255, 0), -1)
        cv2.putText(image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        image_with_grid = draw_grid(image, pool_bbox, rows, cols)

        print(f"Saving output image to {output_path}...")
        cv2.imwrite(output_path, image_with_grid)

        plt.imshow(cv2.cvtColor(image_with_grid, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

        return image_with_grid
    else:
        print("No pool detected in the image.")
        return image

if __name__ == "__main__":
    detect_and_draw_grid('ew.jpg')
