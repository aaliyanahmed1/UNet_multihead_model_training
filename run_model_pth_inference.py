"""
PyTorch Inference Module for COVID-19 Multi-Task Model

This module provides functionality to run inference using the original PyTorch model
for COVID-19 classification and lung segmentation from chest X-ray images.
"""

import os
import argparse
import random
import csv
from typing import List, Tuple

import numpy as np
import torch
import cv2

import model_arch as trainer


def ensure_dir(path: str) -> None:
    """Ensure directory exists, create if it doesn't."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def find_images_in_dataset(root_dir: str, class_names: List[str]) -> List[Tuple[str, str]]:
    """
    Find all images in the dataset directory structure.
    
    Args:
        root_dir (str): Root directory containing class subdirectories
        class_names (List[str]): List of class names to search for
        
    Returns:
        List[Tuple[str, str]]: List of (image_path, class_name) tuples
    """
    samples: List[Tuple[str, str]] = []
    for class_name in class_names:
        images_dir = os.path.join(root_dir, class_name, 'images')
        if not os.path.isdir(images_dir):
            continue
        for fname in os.listdir(images_dir):
            if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                samples.append((os.path.join(images_dir, fname), class_name))
    return samples


def preprocess_image_for_model(image_bgr: np.ndarray, size: Tuple[int, int] = (256, 256)) -> torch.Tensor:
    """
    Preprocess image for model inference.
    
    Args:
        image_bgr (np.ndarray): Input image in BGR format
        size (Tuple[int, int]): Target size for resizing (default: (256, 256))
        
    Returns:
        torch.Tensor: Preprocessed image tensor ready for model input
    """
    image_resized = cv2.resize(image_bgr, (size[0], size[1]))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_norm = image_rgb.astype(np.float32) / 255.0
    chw = np.transpose(image_norm, (2, 0, 1))
    tensor = torch.tensor(chw).unsqueeze(0).float()
    return tensor


def overlay_segmentation(image_bgr: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """
    Overlay segmentation mask on the original image.
    
    Args:
        image_bgr (np.ndarray): Original image in BGR format
        pred_mask (np.ndarray): Predicted segmentation mask
        
    Returns:
        np.ndarray: Image with segmentation overlay
    """
    pred = (pred_mask[0] > 0.5).astype(np.float32)
    overlay = image_bgr.copy().astype(np.float32) / 255.0
    # Red channel for prediction
    overlay[..., 2] = np.maximum(overlay[..., 2], pred)
    overlay = (overlay * 255.0).clip(0, 255).astype(np.uint8)
    return overlay


def annotate_class(overlay_bgr: np.ndarray, text: str) -> np.ndarray:
    """
    Add text annotation to the image.
    
    Args:
        overlay_bgr (np.ndarray): Image with segmentation overlay
        text (str): Text to add to the image
        
    Returns:
        np.ndarray: Annotated image
    """
    out = overlay_bgr.copy()
    cv2.rectangle(out, (5, 5), (5 + 300, 40), (0, 0, 0), -1)
    cv2.putText(out, text, (12, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def run_inference(
    data_dir: str,
    checkpoint_path: str,
    save_dir: str,
    num_samples: int = 12,
    device: str | None = None,
    image_paths: List[str] | None = None
) -> None:
    """
    Run inference on sample images using PyTorch model.
    
    Args:
        data_dir (str): Root directory containing class subfolders with images
        checkpoint_path (str): Path to the trained PyTorch checkpoint
        save_dir (str): Directory to save annotated results and predictions
        num_samples (int): Number of random samples to process (default: 12)
        device (str | None): Device for inference ('cuda' or 'cpu')
        image_paths (List[str] | None): Specific image paths to process
    """
    ensure_dir(save_dir)

    # Device selection
    torch_device = torch.device('cuda' if (device in ['cuda', None] and torch.cuda.is_available()) else 'cpu')

    # Load checkpoint (full dict saved by trainer)
    checkpoint = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)
    class_names = checkpoint.get('class_names', ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia'])

    # Build and load model
    model = trainer.MultiTaskCOVIDModel(num_classes=len(class_names), img_size=(256, 256))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(torch_device)
    model.eval()

    # Gather images
    samples: List[Tuple[str, str]] = []
    if image_paths:
        for p in image_paths:
            samples.append((p, 'unknown'))
    else:
        dataset_samples = find_images_in_dataset(data_dir, class_names)
        if len(dataset_samples) == 0:
            raise RuntimeError('No images found in dataset. Expected structure: <root>/<class>/images/*.png|jpg|jpeg')
        random.seed(42)
        random.shuffle(dataset_samples)
        samples = dataset_samples[:num_samples]

    # CSV for predictions
    csv_path = os.path.join(save_dir, 'inference_predictions.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['image_path', 'true_class', 'pred_class'] + [f'prob_{c}' for c in class_names]
        writer.writerow(header)

        for idx, (img_path, true_cls) in enumerate(samples):
            image_bgr = cv2.imread(img_path)
            if image_bgr is None:
                print(f'Skipping unreadable image: {img_path}')
                continue

            # Keep original-sized for overlay; also prepare model input size
            input_tensor = preprocess_image_for_model(image_bgr, size=(256, 256)).to(torch_device)

            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.softmax(outputs['classification'], dim=1).cpu().numpy()[0]
                pred_idx = int(np.argmax(probs))
                pred_name = class_names[pred_idx]
                seg_mask = outputs['segmentation'].cpu().numpy()[0]

            # Build overlay on resized image for consistency
            resized_for_overlay = cv2.resize(image_bgr, (256, 256))
            overlay = overlay_segmentation(resized_for_overlay, seg_mask)
            prob_str = ', '.join([f'{c}:{probs[i]:.2f}' for i, c in enumerate(class_names)])
            annotated = annotate_class(overlay, f'Pred: {pred_name} | {prob_str}')

            out_name = f'sample_{idx+1}_{os.path.basename(img_path)}'
            out_path = os.path.join(save_dir, out_name)
            cv2.imwrite(out_path, annotated)

            row = [img_path, true_cls, pred_name] + [f'{p:.6f}' for p in probs]
            writer.writerow(row)

    print(f'Inference complete. Images and CSV saved to: {save_dir}')


def parse_args():
    parser = argparse.ArgumentParser(description='Run inference on sample images and save annotated results')
    parser.add_argument('--data_dir', type=str, required=True, help='Dataset root with class subfolders containing images')
    parser.add_argument('--checkpoint', type=str, default=os.path.join('model_checkpoints', 'best_covid_model.pth'), help='Path to trained checkpoint')
    parser.add_argument('--save_dir', type=str, default='output_results', help='Directory to save annotated results')
    parser.add_argument('--num_samples', type=int, default=12, help='Number of random samples from dataset if no image paths provided')
    parser.add_argument('--device', type=str, default=None, help='cuda or cpu (default auto)')
    parser.add_argument('--images', type=str, nargs='*', help='Optional list of image file paths to run inference on')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_inference(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        save_dir=args.save_dir,
        num_samples=args.num_samples,
        device=args.device,
        image_paths=args.images,
    )


