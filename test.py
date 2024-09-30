import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt

def load_custom_model(weights_path='best.pt'):
    repo_dir = 'yolov9'
    model = torch.hub.load(repo_dir, 'custom', path=weights_path, source='local', force_reload=True)
    return model

def draw_grid_in_pool(image, mask, rows, cols):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return image
    pool_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(pool_contour)
    dx, dy = w // cols, h // rows
    for i in range(1, cols):
        cv2.line(image, (x + i * dx, y), (x + i * dx, y + h), (0, 0, 255), 2)
    for i in range(1, rows):
        cv2.line(image, (x, y + i * dy), (x + w, y + i * dy), (0, 0, 255), 2)
    return image

def detect_and_draw_grid(image_path, rows=4, cols=5, output_path='output_image.jpg'):
    model = load_custom_model()
    image = cv2.imread(image_path)
    results = model(image)
    detections = results.xyxy[0].cpu().numpy()
    labels = results.names
    pool_mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        label = labels[int(cls)]
        if label == 'swimming-pool':
            cv2.rectangle(pool_mask, (x1, y1), (x2, y2), 255, -1)
            confidence = conf * 100
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"swimming-pool {confidence:.0f}%"
            (w, h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(image, (x1, y1 - h - 5), (x1 + w, y1), (0, 255, 0), -1)
            cv2.putText(image, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    image_with_grid = draw_grid_in_pool(image, pool_mask, rows, cols)
    cv2.imwrite(output_path, image_with_grid)
    plt.imshow(cv2.cvtColor(image_with_grid, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    return image_with_grid

if __name__ == "__main__":
    detect_and_draw_grid('aq.jpg')
