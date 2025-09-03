import os
import argparse
import json
import random
from typing import List, Tuple

import numpy as np
import cv2
import onnxruntime as ort


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def find_images_in_dataset(root_dir: str, class_names: List[str]) -> List[Tuple[str, str]]:
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
    image_resized = cv2.resize(image_bgr, (size, size))
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    image_norm = image_rgb.astype(np.float32) / 255.0
    chw = np.transpose(image_norm, (2, 0, 1))
    return chw[np.newaxis, ...]


def overlay_segmentation(image_bgr: np.ndarray, seg: np.ndarray) -> np.ndarray:
    pred = (seg[0] > 0.5).astype(np.float32)
    overlay = image_bgr.copy().astype(np.float32) / 255.0
    overlay[..., 2] = np.maximum(overlay[..., 2], pred)
    overlay = (overlay * 255.0).clip(0, 255).astype(np.uint8)
    return overlay


def annotate(overlay_bgr: np.ndarray, text: str) -> np.ndarray:
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
    ensure_dir(save_dir)

    if not os.path.exists(meta_path):
        raise FileNotFoundError(f'Metadata file not found: {meta_path}')
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    class_names: List[str] = meta['class_names']
    img_size: int = int(meta.get('img_size', 256))

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


