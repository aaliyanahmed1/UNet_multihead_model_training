# Multi-Head MONAI Deep Learning Model

A state-of-the-art multi-task deep learning model for COVID-19 detection and lung segmentation from chest X-ray images. This model simultaneously performs classification and segmentation tasks, achieving **94.1% accuracy** on the test dataset.

## Technical Architecture

### Model Structure Overview

The model is implemented using the MONAI framework and follows a multi-head architecture pattern where a shared encoder (UNet backbone) extracts features that are then processed by two specialized heads for different tasks.

```python
class MultiTaskCOVIDModel(nn.Module):
    def __init__(self, num_classes: int = 4, img_size: Tuple[int, int] = (256, 256)):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Shared encoder (UNet backbone)
        self.backbone = UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=64,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
            dropout=0.1
        )
        
        # Dual output heads
        self.seg_head = nn.Sequential(...)      # Segmentation head
        self.classifier = nn.Sequential(...)    # Classification head
```

### Shared Encoder Architecture

The backbone utilizes a modified UNet architecture optimized for medical image analysis:

- **Input Processing**: 3-channel RGB images (256×256 pixels)
- **Encoder Path**: Progressive downsampling with channel expansion
  - Level 1: 32 channels (256×256)
  - Level 2: 64 channels (128×128) 
  - Level 3: 128 channels (64×64)
  - Level 4: 256 channels (32×32)
  - Level 5: 512 channels (16×16)
- **Decoder Path**: Progressive upsampling with channel reduction
- **Skip Connections**: Preserve fine-grained spatial information
- **Residual Units**: 2 residual blocks per level for better gradient flow
- **Normalization**: Batch normalization with 0.1 dropout for regularization

### Dual Output Heads

#### 1. Segmentation Head
```python
self.seg_head = nn.Sequential(
    nn.Conv2d(64, 32, kernel_size=3, padding=1),
    nn.BatchNorm2d(32),
    nn.ReLU(inplace=True),
    nn.Conv2d(32, 1, kernel_size=1),
    nn.Sigmoid()
)
```
- **Purpose**: Generate binary masks for lung region segmentation
- **Architecture**: 64 → 32 → 1 channels
- **Activation**: Sigmoid for probability output (0-1 range)
- **Output**: Binary probability mask (256×256)

#### 2. Classification Head
```python
self.classifier = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(64, 256),
    nn.ReLU(),
    nn.BatchNorm1d(256),
    nn.Dropout(0.3),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.BatchNorm1d(128),
    nn.Dropout(0.2),
    nn.Linear(128, num_classes)
)
```
- **Purpose**: Classify chest X-ray images into 4 categories
- **Architecture**: Global Average Pooling → 64 → 256 → 128 → 4
- **Regularization**: Progressive dropout (0.5, 0.3, 0.2)
- **Output**: Logits for 4 classes (COVID, Lung_Opacity, Normal, Viral Pneumonia)

### Multi-Task Loss Function

The model employs a sophisticated loss combination that balances both tasks:

```python
class MultiTaskLoss(nn.Module):
    def __init__(self, seg_weight: float = 1.0, cls_weight: float = 2.0):
        super().__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        
        # Segmentation losses
        self.dice_loss = DiceLoss(sigmoid=False, squared_pred=True)
        self.bce_loss = nn.BCELoss()
        
        # Classification loss
        self.cls_loss = nn.CrossEntropyLoss()
    
    def forward(self, outputs, targets):
        # Combined segmentation loss
        dice_loss = self.dice_loss(outputs['segmentation'], targets['mask'])
        bce_loss = self.bce_loss(outputs['segmentation'], targets['mask'])
        seg_loss = dice_loss + bce_loss
        
        # Classification loss
        cls_loss = self.cls_loss(outputs['classification'], targets['class'])
        
        # Weighted combination
        total_loss = self.seg_weight * seg_loss + self.cls_weight * cls_loss
        return {'total_loss': total_loss, 'seg_loss': seg_loss, 'cls_loss': cls_loss}
```

**Loss Components:**
- **Dice Loss**: Handles class imbalance in segmentation masks
- **Binary Cross-Entropy**: Provides pixel-wise classification loss
- **Cross-Entropy**: Standard classification loss with class weighting
- **Weighting**: Segmentation (1.0) + Classification (2.0) for balanced learning

## Dataset Preprocessing and Structure

### Expected Dataset Organization

The preprocessing pipeline expects the following directory structure:

```
data_dir/
├── COVID/
│   ├── images/          # COVID X-ray images (.png, .jpg, .jpeg)
│   └── masks/           # Corresponding segmentation masks
├── Lung_Opacity/
│   ├── images/          # Lung opacity X-ray images
│   └── masks/           # Corresponding segmentation masks
├── Normal/
│   ├── images/          # Normal X-ray images
│   └── masks/           # Corresponding segmentation masks
└── Viral Pneumonia/
    ├── images/          # Viral pneumonia X-ray images
    └── masks/           # Corresponding segmentation masks
```

### Data Collection and Preprocessing Pipeline

#### 1. File Discovery and Organization
```python
def prepare_dataset_splits(data_dir: str, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15):
    class_names = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    all_data = []
    for class_name in class_names:
        images_path = os.path.join(data_dir, class_name, 'images')
        masks_path = os.path.join(data_dir, class_name, 'masks')
        
        for img_file in os.listdir(images_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Find corresponding mask file
                mask_file = find_corresponding_mask(img_file, masks_path)
                all_data.append({
                    'image': img_path,
                    'mask': mask_file,
                    'class': class_to_idx[class_name],
                    'class_name': class_name
                })
```

#### 2. Stratified Data Splitting
The code implements **stratified splitting** to maintain class distribution across all splits:

```python
# First split: Train vs (Val + Test)
train_data, temp_data, train_labels, temp_labels = train_test_split(
    all_data, labels,
    test_size=(val_ratio + test_ratio),  # 0.3 (15% + 15%)
    random_state=42,
    stratify=labels  # Maintains class proportions
)

# Second split: Val vs Test
val_test_ratio = val_ratio / (val_ratio + test_ratio)  # 0.5
val_data, test_data, _, _ = train_test_split(
    temp_data, temp_labels,
    test_size=(1 - val_test_ratio),  # 0.5
    random_state=42,
    stratify=temp_labels
)
```

**Final Split Distribution:**
- **Training**: 70% of total data
- **Validation**: 15% of total data  
- **Test**: 15% of total data

#### 3. Image Preprocessing Pipeline
```python
class COVID19Dataset(Dataset):
    def __getitem__(self, idx):
        # Load and convert image
        image = Image.open(sample['image']).convert('RGB')
        image = np.array(image).astype(np.float32)
        
        # Resize to standard size
        image = cv2.resize(image, (256, 256))
        
        # Normalize to [0, 1] range
        image = image / 255.0
        
        # Convert to tensor (CHW format)
        image = torch.tensor(image).permute(2, 0, 1).float()
```

#### 4. Mask Preprocessing
```python
# Load mask (if exists)
if sample['mask'] and os.path.exists(sample['mask']):
    mask = Image.open(sample['mask']).convert('L')  # Grayscale
    mask = np.array(mask).astype(np.float32)
    # Binary threshold (values > 127 become 1, others become 0)
    mask = (mask > 127).astype(np.float32)
else:
    # Create empty mask if not available
    mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

# Resize mask to match image
mask = cv2.resize(mask, (256, 256))

# Add channel dimension for tensor
mask = torch.tensor(mask).unsqueeze(0).float()
```

#### 5. Data Augmentation (Training Only)
The model applies augmentations only during training to prevent overfitting:

```python
if hasattr(self, 'is_training') and self.is_training:
    # Random horizontal flip (50% probability)
    if np.random.random() > 0.5:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    
    # Random rotation (30% probability, ±10 degrees)
    if np.random.random() > 0.7:
        angle = np.random.uniform(-10, 10)
        center = (128, 128)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, M, (256, 256))
        mask = cv2.warpAffine(mask, M, (256, 256))
    
    # Add Gaussian noise (50% probability)
    if np.random.random() > 0.5:
        noise = np.random.normal(0, 5, image.shape)
        image = np.clip(image + noise, 0, 255)
```

### Key Preprocessing Features

1. **Stratified Splitting**: Maintains class balance across all splits
2. **Flexible Mask Handling**: Works with or without segmentation masks
3. **Standardized Input**: All images resized to 256×256 pixels
4. **Proper Normalization**: Images normalized to [0,1] range
5. **Data Augmentation**: Applied only during training to prevent overfitting
6. **Consistent Format**: All data converted to PyTorch tensors in CHW format
7. **Binary Masks**: Segmentation masks converted to binary (0/1) values

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
