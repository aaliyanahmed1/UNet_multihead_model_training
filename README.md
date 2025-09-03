# COVID-19 Multi-Task Deep Learning Model

A state-of-the-art multi-task deep learning model for COVID-19 detection and lung segmentation from chest X-ray images. This model simultaneously performs classification and segmentation tasks, achieving **94.1% accuracy** on the test dataset.

## Model Architecture

The model employs a sophisticated multi-task architecture built on a UNet backbone with dual output heads:

### Shared Encoder (UNet Backbone)
- **Architecture**: Modified UNet with enhanced feature extraction capabilities
- **Input**: RGB chest X-ray images (256×256 pixels)
- **Channels**: (32, 64, 128, 256, 512) with residual units
- **Normalization**: Batch normalization with dropout (0.1)
- **Output Features**: 64-channel feature maps

### Dual Output Heads

#### 1. Classification Head
- **Purpose**: Classify chest X-ray images into 4 categories
- **Architecture**: 
  - Global Average Pooling → 64 features
  - Fully connected layers: 64 → 256 → 128 → 4
  - Dropout layers (0.5, 0.3, 0.2) for regularization
  - Batch normalization for stable training
- **Output**: Logits for 4 classes

#### 2. Segmentation Head
- **Purpose**: Generate binary masks for lung region segmentation
- **Architecture**:
  - Convolutional layers: 64 → 32 → 1 channels
  - Batch normalization and ReLU activation
  - Sigmoid activation for probability output
- **Output**: Binary probability mask (256×256)

### Multi-Task Loss Function
The model uses a combined loss function that balances both tasks:
- **Segmentation Loss**: Dice Loss + Binary Cross-Entropy Loss
- **Classification Loss**: Cross-Entropy Loss with class weighting
- **Weighting**: Segmentation (1.0) + Classification (2.0)

## Dataset

The model was trained on the [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database) from Kaggle, a comprehensive chest X-ray dataset containing **21,165 samples** across 4 classes:

| Class | Training Samples | Validation Samples | Test Samples | Total |
|-------|------------------|-------------------|--------------|-------|
| COVID | 2,531 | 543 | 542 | 3,616 |
| Lung_Opacity | 4,208 | 902 | 902 | 6,012 |
| Normal | 7,134 | 1,529 | 1,529 | 10,192 |
| Viral Pneumonia | 942 | 201 | 202 | 1,345 |

### Dataset Structure
The COVID-19 Radiography Database provides:
- **Images**: High-quality chest X-ray images in PNG/JPG format
- **Masks**: Corresponding binary segmentation masks for lung regions
- **Format**: RGB images normalized to [0,1] range
- **Size**: Resized to 256×256 pixels for training
- **Source**: Curated collection of chest X-ray images from various medical institutions
- **Quality**: Professional medical imaging data with expert annotations

### Dataset Details
- **Total Images**: 21,165 chest X-ray images
- **Classes**: 4 distinct categories (COVID, Lung Opacity, Normal, Viral Pneumonia)
- **Segmentation Masks**: Binary masks highlighting lung regions for each image
- **Resolution**: Original images resized to 256×256 pixels for model training
- **License**: Available under appropriate medical data usage terms
- **Citation**: Please cite the original dataset when using this model

## Model Performance

### Test Set Results (Final Evaluation)
- **Overall Accuracy**: **94.1%**
- **Macro Average F1-Score**: 94.6%
- **Weighted Average F1-Score**: 94.1%

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| COVID | 98.2% | 93.0% | 95.5% | 542 |
| Lung_Opacity | 92.7% | 90.1% | 91.4% | 902 |
| Normal | 93.1% | 96.8% | 95.0% | 1,529 |
| Viral Pneumonia | 98.5% | 95.0% | 96.7% | 202 |

### Evaluation Metrics
The model was evaluated using comprehensive metrics:

#### Classification Metrics
- **Accuracy**: Overall classification accuracy
- **Precision**: True positive rate for each class
- **Recall**: Sensitivity for each class
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Confusion Matrix**: Detailed classification breakdown

#### Segmentation Metrics
- **Dice Coefficient**: Overlap between predicted and ground truth masks
- **IoU (Intersection over Union)**: Spatial overlap metric
- **Binary Cross-Entropy**: Pixel-wise classification loss

## Model Export and Deployment

### ONNX Export (`export_to_onnx.py`)

The trained PyTorch model is exported to ONNX format for cross-platform deployment:

```python
def export_to_onnx(
    checkpoint_path: str,
    onnx_path: str,
    img_size: int = 256,
    opset: int = 17,
    dynamic_batch: bool = True,
    device: str | None = None
) -> None:
    """Export trained PyTorch model to ONNX format for deployment.
    
    This function loads a trained checkpoint, wraps the model for ONNX export,
    and saves both the ONNX model and metadata for inference.
    
    Args:
        checkpoint_path (str): Path to the trained PyTorch checkpoint file.
        onnx_path (str): Output path for the ONNX model file.
        img_size (int): Image size used for model input (default: 256).
        opset (int): ONNX opset version for compatibility (default: 17).
        dynamic_batch (bool): Enable dynamic batch size for flexible inference.
        device (str | None): Device to use for export ('cuda' or 'cpu').
    
    Returns:
        None: Saves ONNX model and metadata files.
    
    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
        RuntimeError: If ONNX export fails.
    
    Example:
        export_to_onnx(
            checkpoint_path='model_checkpoints/best_covid_model.pth',
            onnx_path='models/covid_multitask.onnx',
            img_size=256,
            opset=17
        )
    """
```

**Key Features:**
- **Dynamic Batch Support**: Enables variable batch sizes during inference
- **Metadata Export**: Saves model configuration and class information
- **ONNX Validation**: Automatic model validation after export
- **Cross-Platform**: Compatible with ONNX Runtime on various platforms

### ONNX Model Metadata
The exported model includes comprehensive metadata:

```json
{
  "class_names": ["COVID", "Lung_Opacity", "Normal", "Viral Pneumonia"],
  "img_size": 256,
  "normalization": "RGB, scaled to [0,1]",
  "outputs": {
    "segmentation": "Sigmoid probability mask (B,1,H,W)",
    "classification": "Logits (B,C)"
  },
  "opset": 17,
  "dynamic_batch": true
}
```

## Inference Scripts

### PyTorch Inference (`run_model_pth_inference.py`)

Runs inference using the original PyTorch model:

```python
def run_inference(
    data_dir: str,
    checkpoint_path: str,
    save_dir: str,
    num_samples: int = 12,
    device: str | None = None,
    image_paths: List[str] | None = None
) -> None:
    """Run inference on sample images using PyTorch model.
    
    Performs classification and segmentation on chest X-ray images,
    generating annotated visualizations and prediction results.
    
    Args:
        data_dir (str): Root directory containing class subfolders with images.
        checkpoint_path (str): Path to the trained PyTorch checkpoint.
        save_dir (str): Directory to save annotated results and predictions.
        num_samples (int): Number of random samples to process (default: 12).
        device (str | None): Device for inference ('cuda' or 'cpu').
        image_paths (List[str] | None): Specific image paths to process.
    
    Returns:
        None: Saves annotated images and CSV predictions.
    
    Raises:
        RuntimeError: If no images found in dataset directory.
        FileNotFoundError: If checkpoint file doesn't exist.
    
    Example:
        run_inference(
            data_dir='dataset/',
            checkpoint_path='model_checkpoints/best_covid_model.pth',
            save_dir='output_results/',
            num_samples=12
        )
    """
```

**Features:**
- **Visual Annotations**: Overlays segmentation masks on original images
- **Probability Display**: Shows classification probabilities for all classes
- **CSV Export**: Saves detailed predictions in CSV format
- **Batch Processing**: Handles multiple images efficiently

### ONNX Inference (`run_onnx_inference.py`)

Runs inference using the exported ONNX model:

```python
def run(
    data_dir: str,
    onnx_path: str,
    meta_path: str,
    save_dir: str,
    num_samples: int,
    images: List[str] | None = None,
    providers: List[str] | None = None
) -> None:
    """Run inference using ONNX Runtime for optimized performance.
    
    Performs fast inference using ONNX Runtime with support for
    multiple execution providers (CPU, CUDA, etc.).
    
    Args:
        data_dir (str): Root directory containing class subfolders.
        onnx_path (str): Path to the exported ONNX model file.
        meta_path (str): Path to the ONNX metadata JSON file.
        save_dir (str): Directory to save inference results.
        num_samples (int): Number of samples to process.
        images (List[str] | None): Specific image paths to process.
        providers (List[str] | None): ONNX Runtime execution providers.
    
    Returns:
        None: Saves annotated results and predictions.
    
    Raises:
        FileNotFoundError: If ONNX model or metadata files don't exist.
        RuntimeError: If no images found in dataset directory.
    
    Example:
        run(
            data_dir='dataset/',
            onnx_path='models/covid_multitask.onnx',
            meta_path='models/covid_multitask.onnx.meta.json',
            save_dir='onnxoutputs/',
            num_samples=12,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
    """
```

**Advantages:**
- **Optimized Performance**: Faster inference compared to PyTorch
- **Cross-Platform**: Works on various hardware and operating systems
- **Multiple Providers**: Supports CPU, CUDA, and other execution providers
- **Memory Efficient**: Lower memory footprint during inference

## Usage Examples

### Training the Model
```bash
python model_arch.py --data_dir /path/to/dataset --num_epochs 100 --batch_size 16
```

### Exporting to ONNX
```bash
python export_to_onnx.py --checkpoint model_checkpoints/best_covid_model.pth --onnx_path models/covid_multitask.onnx
```

### PyTorch Inference
```bash
python run_model_pth_inference.py --data_dir /path/to/dataset --checkpoint model_checkpoints/best_covid_model.pth --save_dir output_results
```

### ONNX Inference
```bash
python run_onnx_inference.py --data_dir /path/to/dataset --onnx_path models/covid_multitask.onnx --save_dir onnxoutputs --use_cuda
```

## File Structure

```
lightining_work/
├── model_arch.py                    # Main training script with model architecture
├── export_to_onnx.py               # ONNX export functionality
├── run_model_pth_inference.py      # PyTorch inference script
├── run_onnx_inference.py           # ONNX inference script
├── model_checkpoints/              # Training checkpoints and results
│   ├── best_covid_model.pth        # Best performing model
│   ├── training_curves.png         # Training progress visualization
│   └── training_summary.json       # Training metrics summary
├── models/                         # Exported models
│   ├── covid_multitask.onnx        # ONNX model file
│   └── covid_multitask.onnx.meta.json  # Model metadata
├── output_results/                 # PyTorch inference results
├── onnxoutputs/                    # ONNX inference results
└── test_results/                   # Final evaluation results
    ├── classification_report.json  # Detailed performance metrics
    ├── confusion_matrix_test.png   # Confusion matrix visualization
    └── roc_curves_test.png         # ROC curves for all classes
```

## Dependencies

### Core Requirements
- **PyTorch**: Deep learning framework
- **MONAI**: Medical imaging AI toolkit
- **OpenCV**: Image processing
- **NumPy**: Numerical computations
- **PIL**: Image handling
- **scikit-learn**: Machine learning utilities
- **matplotlib**: Visualization
- **tqdm**: Progress bars

### ONNX Requirements
- **ONNX Runtime**: Cross-platform inference engine
- **ONNX**: Model format specification

## Technical Specifications

- **Input Format**: RGB images, 256×256 pixels, normalized to [0,1]
- **Model Size**: ~15M parameters
- **Training Time**: ~4-6 hours on GPU (100 epochs)
- **Inference Speed**: ~50ms per image (GPU), ~200ms per image (CPU)
- **Memory Usage**: ~2GB GPU memory during training, ~500MB during inference

## Performance Highlights

- **94.1% Test Accuracy**: Exceptional performance on unseen data
- **Multi-Task Learning**: Simultaneous classification and segmentation
- **Robust Architecture**: Handles class imbalance effectively
- **Production Ready**: ONNX export for deployment
- **Comprehensive Evaluation**: Detailed metrics across all classes

This model represents a significant advancement in automated COVID-19 detection from chest X-rays, providing both diagnostic classification and anatomical segmentation capabilities in a single, efficient architecture.

## Dataset Citation

If you use this model in your research, please cite the original dataset:

```
COVID-19 Radiography Database
https://www.kaggle.com/datasets/tawsifurrahman/covid19-radiography-database
```

The COVID-19 Radiography Database is a valuable resource for medical AI research, providing high-quality chest X-ray images with expert annotations for both classification and segmentation tasks.
