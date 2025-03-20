import cv2
import torch
from pathlib import Path
import numpy as np
from ultralytics import YOLO
import os

def detect_and_save(model, image_path, save_path, conf_thres=0.25):
    # Read image
    img = cv2.imread(str(image_path))
    orig_img = img.copy()

    # Perform inference
    results = model(img)[0]  # Get first result

    # Process detections
    for det in results.boxes.data:  # Access boxes directly
        if det[4] >= conf_thres:  # Check confidence threshold
            x1, y1, x2, y2, conf = det[:5].cpu().numpy()
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            
            # Draw bounding box
            cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Add confidence score text
            conf_text = f'palm_tree_{conf:.2f}'
            cv2.putText(orig_img, conf_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3)

    # Save the image
    cv2.imwrite(str(save_path), orig_img)

def process_images(model, input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    for img_path in input_path.glob('*.jpg'):
        save_path = output_path / f'detected_{img_path.name}'
        detect_and_save(model, img_path, save_path)


if __name__ == "__main__":
    # Usage example:
    model = YOLO('runs/detect/palm_tree_detection/weights/best.pt')  # adjust path to your weights
    test_dir = 'dataset/yolo/images/test'
    save_dir = 'output'
    if not os.path.exists(save_dir):
        os.makedirs
    process_images(model, test_dir, save_dir)
