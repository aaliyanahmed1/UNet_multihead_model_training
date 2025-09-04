"""
ONNX Inference Module for COVID-19 Multi-Task Model

This module provides functionality to run inference using the exported ONNX model
for COVID-19 classification and lung segmentation from chest X-ray images.
"""

import os
import argparse
import json
import random
from typing import List, Tuple, Optional

import numpy as np
import cv2
import onnxruntime as ort


def is_xray_like(image_bgr: Optional[np.ndarray]) -> Tuple[bool, str]:
    """Validate if image appears to be a chest X-ray with balanced criteria.
    
    Args:
        image_bgr: Input BGR image or None
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if image_bgr is None:
        return False, "Failed to read the image file."
    
    # Check image dimensions
    h, w = image_bgr.shape[:2]
    if min(h, w) < 256:
        return False, "Image is too small. Minimum dimension should be 256 pixels for X-ray analysis."
    
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Check if image is actually grayscale (not just black and white)
    # Calculate color channel differences
    b, g, r = cv2.split(image_bgr.astype(np.float32))
    diff_bg = np.mean(np.abs(b - g))
    diff_br = np.mean(np.abs(b - r))
    diff_gr = np.mean(np.abs(g - r))
    avg_diff = (diff_bg + diff_br + diff_gr) / 3.0
    
    if avg_diff > 8.0:  # Increased from 5.0 to allow for slight color variations
        return False, "Image appears to be colorful. X-ray images must be grayscale."
    
    # 2. Check for X-ray specific characteristics with more lenient ranges
    # X-rays should have specific intensity distribution
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    
    # More lenient brightness range for different X-ray types
    if mean_intensity < 20 or mean_intensity > 240:  # Widened from 30-225
        return False, f"Image brightness ({mean_intensity:.1f}) is outside X-ray range (20-240)."
    
    if std_intensity < 8:  # Reduced from 15 to allow lighter X-rays
        return False, f"Image contrast ({std_intensity:.1f}) is too low for X-ray. X-rays need some contrast."
    
    # 3. Check for X-ray anatomical features with lower threshold
    # X-rays should have some lung-like patterns (not just uniform)
    # Calculate local variance to detect anatomical structures
    kernel = np.ones((8, 8), np.float32) / 64
    local_mean = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    local_variance = cv2.filter2D((gray.astype(np.float32) - local_mean)**2, -1, kernel)
    avg_local_variance = np.mean(local_variance)
    
    if avg_local_variance < 20:  # Reduced from 50 to allow lighter X-rays
        return False, "Image lacks anatomical detail. X-rays should show lung structures and ribs."
    
    # 4. Check for X-ray specific intensity patterns with lower entropy threshold
    # X-rays have characteristic histogram distribution
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_norm = hist.flatten() / np.sum(hist)
    
    # Check if histogram has proper X-ray distribution (not too uniform, not too peaked)
    entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-10))
    if entropy < 3.0:  # Reduced from 4.0 to allow lighter X-rays
        return False, "Image histogram is too uniform. X-rays have characteristic intensity distribution."
    
    # 5. Check for rib-like structures with lower threshold
    # Apply edge detection to find potential rib structures
    edges = cv2.Canny(gray, 30, 100)  # Lowered thresholds for lighter X-rays
    horizontal_kernel = np.array([[1, 1, 1, 1, 1]], np.uint8)
    horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
    
    # Count horizontal line pixels
    horizontal_pixels = np.sum(horizontal_lines > 0)
    total_pixels = gray.size
    
    if horizontal_pixels / total_pixels < 0.0005:  # Reduced from 0.001
        return False, "No rib-like structures detected. X-rays should show rib outlines."
    
    # 6. Check for lung field characteristics with more lenient thresholds
    # X-rays should have darker lung areas and brighter bone areas
    # Calculate ratio of dark to bright areas
    dark_threshold = np.percentile(gray, 25)  # Changed from 30
    bright_threshold = np.percentile(gray, 75)  # Changed from 70
    
    dark_pixels = np.sum(gray < dark_threshold)
    bright_pixels = np.sum(gray > bright_threshold)
    
    if dark_pixels / total_pixels < 0.05 or bright_pixels / total_pixels < 0.05:  # Reduced from 0.1
        return False, "Image lacks proper X-ray contrast between lung fields and bones."
    
    # 7. Final check: Ensure it's not just a simple black and white image
    # Check for intermediate gray values (X-rays have many gray levels)
    unique_values = len(np.unique(gray))
    if unique_values < 50:  # Reduced from 100 to allow lighter X-rays
        return False, f"Image has too few gray levels ({unique_values}). X-rays have continuous gray scale."
    
    return True, ""


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


def preprocess(image_bgr: np.ndarray, size: int) -> np.ndarray:
    """
    Preprocess image for ONNX model inference.
    
    Args:
        image_bgr (np.ndarray): Input image in BGR format
        size (int): Target size for resizing
        
    Returns:
        np.ndarray: Preprocessed image array ready for ONNX model input
    """
    image_resized = cv2.resize(image_bgr, (size, size))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_norm = image_rgb.astype(np.float32) / 255.0
    chw = np.transpose(image_norm, (2, 0, 1))
    return chw[np.newaxis, ...]


def overlay_segmentation(image_bgr: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """
    Overlay segmentation mask on the original image.
    
    Args:
        image_bgr (np.ndarray): Original image in BGR format
        seg (np.ndarray): Predicted segmentation mask
        
    Returns:
        np.ndarray: Image with segmentation overlay
    """
    pred = (seg[0] > 0.5).astype(np.float32)
    overlay = image_bgr.copy().astype(np.float32) / 255.0
    overlay[..., 2] = np.maximum(overlay[..., 2], pred)
    overlay = (overlay * 255.0).clip(0, 255).astype(np.uint8)
    return overlay


def annotate(overlay_bgr: np.ndarray, text: str) -> np.ndarray:
    """
    Add text annotation to the image.
    
    Args:
        overlay_bgr (np.ndarray): Image with segmentation overlay
        text (str): Text to add to the image
        
    Returns:
        np.ndarray: Annotated image
    """
    out = overlay_bgr.copy()
    cv2.rectangle(out, (5, 5), (5 + 380, 40), (0, 0, 0), -1)
    cv2.putText(out, text, (12, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def run(
    data_dir: str,
    onnx_path: str,
    meta_path: str,
    save_dir: str,
    num_samples: int,
    images: List[str] | None = None,
    providers: List[str] | None = None
) -> None:
    """
    Run inference using ONNX Runtime for optimized performance.
    
    Args:
        data_dir (str): Root directory containing class subfolders
        onnx_path (str): Path to the exported ONNX model file
        meta_path (str): Path to the ONNX metadata JSON file
        save_dir (str): Directory to save inference results
        num_samples (int): Number of samples to process
        images (List[str] | None): Specific image paths to process
        providers (List[str] | None): ONNX Runtime execution providers
    """
    ensure_dir(save_dir)

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f'Metadata file not found: {meta_path}')
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    class_names: List[str] = meta['class_names']
    img_size: int = int(meta.get('img_size', 256))

    # Initialize ONNX Runtime session
    sess_options = ort.SessionOptions()
    session = ort.InferenceSession(onnx_path, sess_options, providers=providers or ['CPUExecutionProvider'])

    input_name = session.get_inputs()[0].name
    out_names = [o.name for o in session.get_outputs()]

    # Gather images
    if images:
        samples = [(p, 'unknown') for p in images]
    else:
        dataset_samples = find_images_in_dataset(data_dir, class_names)
        if len(dataset_samples) == 0:
            raise RuntimeError('No images found in dataset. Expected structure: <root>/<class>/images/*.png|jpg|jpeg')
        random.seed(42)
        random.shuffle(dataset_samples)
        samples = dataset_samples[:num_samples]

    for idx, (img_path, true_cls) in enumerate(samples):
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            print(f'Skipping unreadable image: {img_path}')
            continue

        # Validate if image is X-ray like
        is_valid, error_msg = is_xray_like(image_bgr)
        if not is_valid:
            print(f'Invalid X-ray image: {img_path} - {error_msg}')
            continue

        input_data = preprocess(image_bgr, img_size)

        seg, logits = session.run(out_names, {input_name: input_data})
        probs = (np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True))[0]
        pred_idx = int(np.argmax(probs))
        pred_name = class_names[pred_idx]

        resized = cv2.resize(image_bgr, (img_size, img_size))
        overlay = overlay_segmentation(resized, seg[0])
        prob_str = ', '.join([f'{c}:{probs[i]:.2f}' for i, c in enumerate(class_names)])
        annotated = annotate(overlay, f'Pred: {pred_name} | {prob_str}')

        out_name = f'onnx_{idx+1}_{os.path.basename(img_path)}'
        out_path = os.path.join(save_dir, out_name)
        cv2.imwrite(out_path, annotated)

    print(f'ONNX inference complete. Results saved to: {save_dir}')


def parse_args():
    parser = argparse.ArgumentParser(description='Run ONNX inference for multitask COVID model')
    parser.add_argument('--data_dir', type=str, required=True, help='Dataset root with class subfolders containing images')
    parser.add_argument('--onnx_path', type=str, default=os.path.join('output_results', 'covid_multitask.onnx'), help='Path to exported ONNX model')
    parser.add_argument('--meta_path', type=str, default=os.path.join('output_results', 'covid_multitask.onnx.meta.json'), help='Path to ONNX metadata JSON')
    parser.add_argument('--save_dir', type=str, default='onnxoutputs', help='Directory to save annotated results')
    parser.add_argument('--num_samples', type=int, default=12, help='Number of random samples if no images provided')
    parser.add_argument('--images', type=str, nargs='*', help='Optional list of image file paths')
    parser.add_argument('--use_cuda', action='store_true', help='Use CUDAExecutionProvider if available')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if args.use_cuda else ['CPUExecutionProvider']
    run(
        data_dir=args.data_dir,
        onnx_path=args.onnx_path,
        meta_path=args.meta_path,
        save_dir=args.save_dir,
        num_samples=args.num_samples,
        images=args.images,
        providers=providers,
    )


