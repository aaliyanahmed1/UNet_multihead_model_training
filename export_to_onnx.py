"""
ONNX Export Module for COVID-19 Multi-Task Model

This module provides functionality to export trained PyTorch models to ONNX format
for cross-platform deployment and optimized inference.
"""

import os
import json
import argparse
import torch
import torch.nn as nn

import model_arch as trainer


class ModelWrapper(nn.Module):
    """
    Wrapper class for ONNX export compatibility.
    
    ONNX export requires models to return tuples instead of dictionaries,
    so this wrapper converts the model's dictionary output to a tuple format.
    """
    
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        """
        Forward pass that returns tuple for ONNX compatibility.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            Tuple: (segmentation_output, classification_output)
        """
        outputs = self.base_model(x)
        # Return a tuple for ONNX export: (segmentation, classification)
        return outputs['segmentation'], outputs['classification']


def export_to_onnx(
    checkpoint_path: str,
    onnx_path: str,
    img_size: int = 256,
    opset: int = 17,
    dynamic_batch: bool = True,
    device: str | None = None
) -> None:
    """
    Export trained PyTorch model to ONNX format for deployment.
    
    Args:
        checkpoint_path (str): Path to the trained PyTorch checkpoint file
        onnx_path (str): Output path for the ONNX model file
        img_size (int): Image size used for model input (default: 256)
        opset (int): ONNX opset version for compatibility (default: 17)
        dynamic_batch (bool): Enable dynamic batch size for flexible inference
        device (str | None): Device to use for export ('cuda' or 'cpu')
    """
    # Device selection
    torch_device = torch.device('cuda' if (device in ['cuda', None] and torch.cuda.is_available()) else 'cpu')

    # Load checkpoint (full dict saved by trainer)
    checkpoint = torch.load(checkpoint_path, map_location=torch_device, weights_only=False)
    class_names = checkpoint.get('class_names', ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia'])

    # Build model and load weights
    model = trainer.MultiTaskCOVIDModel(num_classes=len(class_names), img_size=(img_size, img_size))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(torch_device)
    model.eval()

    # Wrap model for ONNX export
    wrapped = ModelWrapper(model).to(torch_device).eval()

    # Create dummy input for ONNX export
    dummy = torch.randn(1, 3, img_size, img_size, device=torch_device)

    # Prepare output directories
    os.makedirs(os.path.dirname(onnx_path) or '.', exist_ok=True)

    # Configure dynamic axes for variable batch sizes
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            'input': {0: 'batch'},
            'segmentation': {0: 'batch'},
            'classification': {0: 'batch'}
        }

    # Export model to ONNX format
    torch.onnx.export(
        wrapped,
        dummy,
        onnx_path,
        input_names=['input'],
        output_names=['segmentation', 'classification'],
        opset_version=opset,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True
    )

    # Save metadata alongside the ONNX model
    meta = {
        'class_names': class_names,
        'img_size': img_size,
        'normalization': 'RGB, scaled to [0,1]',
        'outputs': {
            'segmentation': 'Sigmoid probability mask (B,1,H,W)',
            'classification': 'Logits (B,C)'
        },
        'opset': opset,
        'dynamic_batch': dynamic_batch
    }
    with open(onnx_path + '.meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    # Optional ONNX validation
    try:
        import onnx
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
    except Exception as e:
        print(f"Warning: ONNX check skipped or failed: {e}")

    print(f'Exported ONNX model to: {onnx_path}')
    print(f'Metadata saved to: {onnx_path}.meta.json')


def parse_args():
    """Parse command line arguments for ONNX export."""
    parser = argparse.ArgumentParser(description='Export trained PyTorch model to ONNX')
    parser.add_argument('--checkpoint', type=str, default=os.path.join('model_checkpoints', 'best_covid_model.pth'), help='Path to trained checkpoint')
    parser.add_argument('--onnx_path', type=str, default=os.path.join('output_results', 'covid_multitask.onnx'), help='Output ONNX file path')
    parser.add_argument('--img_size', type=int, default=256, help='Image size used for export (square)')
    parser.add_argument('--opset', type=int, default=17, help='ONNX opset version')
    parser.add_argument('--no_dynamic_batch', action='store_true', help='Disable dynamic batch dimension')
    parser.add_argument('--device', type=str, default=None, help='cuda or cpu (default auto)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    export_to_onnx(
        checkpoint_path=args.checkpoint,
        onnx_path=args.onnx_path,
        img_size=args.img_size,
        opset=args.opset,
        dynamic_batch=not args.no_dynamic_batch,
        device=args.device
    )


