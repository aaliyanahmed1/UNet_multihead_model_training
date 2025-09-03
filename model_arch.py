# =============================================================================
# COVID-19 Multi-Task Deep Learning Model
# =============================================================================
# This module implements a sophisticated multi-task deep learning model for
# COVID-19 detection and lung segmentation from chest X-ray images.
# 
# Key Features:
# - Multi-task learning: simultaneous classification and segmentation
# - UNet backbone with dual output heads
# - Comprehensive data preprocessing pipeline
# - Stratified train/validation/test splitting
# - Advanced loss function combining Dice and Cross-Entropy losses
# - Full training pipeline with metrics tracking and visualization
# =============================================================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import json
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

# MONAI (Medical Open Network for AI) imports for medical image processing
import monai
from monai.networks.nets import UNet  # U-Net architecture for medical segmentation
from monai.networks.layers import Norm  # Normalization layers
from monai.losses import DiceLoss, DiceCELoss  # Dice loss for segmentation
from monai.metrics import DiceMetric  # Dice coefficient metric
from monai.transforms import (  # Data augmentation transforms
    Compose, LoadImage, AddChannel, ScaleIntensity, RandRotate90,
    RandFlip, RandGaussianNoise, Resize, ToTensor, RandAffine,
    RandGaussianSmooth, RandAdjustContrast
)

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
# Configure logging for training progress tracking and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============================================================================
# CUSTOM DATASET CLASS
# =============================================================================
class COVID19Dataset(Dataset):
    """
    Custom PyTorch Dataset for COVID-19 chest X-ray classification and segmentation.
    
    This dataset handles:
    - Loading chest X-ray images and corresponding segmentation masks
    - Image preprocessing (resizing, normalization)
    - Data augmentation during training
    - Proper tensor conversion for PyTorch models
    
    Args:
        data_list (List[Dict]): List of dictionaries containing image paths, mask paths,
                               class labels, and class names
        transform (bool, optional): Whether to apply data augmentation transforms
    
    Attributes:
        data_list (List[Dict]): List of data samples
        transform (bool): Flag for applying transforms
        class_names (List[str]): List of class names
        class_to_idx (Dict[str, int]): Mapping from class names to indices
    """
    
    def __init__(self, data_list: List[Dict], transform=None):
        self.data_list = data_list
        self.transform = transform
        
        # Define class mapping for 4-class COVID-19 classification
        self.class_names = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
        self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        
    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.data_list)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            Dict: Dictionary containing:
                - 'image': Preprocessed image tensor (3, 256, 256)
                - 'mask': Preprocessed mask tensor (1, 256, 256)
                - 'class': Class label tensor
                - 'image_path': Original image path
                - 'class_name': Class name string
        """
        sample = self.data_list[idx]
        
        # =====================================================================
        # IMAGE LOADING AND PREPROCESSING
        # =====================================================================
        # Load image and convert to RGB format for consistent processing
        image = Image.open(sample['image']).convert('RGB')
        image = np.array(image).astype(np.float32)  # Convert to float32 for processing
        
        # =====================================================================
        # MASK LOADING AND PREPROCESSING
        # =====================================================================
        # Load segmentation mask if available, otherwise create empty mask
        if sample['mask'] and os.path.exists(sample['mask']):
            # Load mask as grayscale image
            mask = Image.open(sample['mask']).convert('L')
            mask = np.array(mask).astype(np.float32)
            # Apply binary threshold: values > 127 become 1, others become 0
            # This converts grayscale masks to binary segmentation masks
            mask = (mask > 127).astype(np.float32)
        else:
            # Create empty mask if no mask file is available
            # This handles cases where segmentation masks are missing
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        
        # =====================================================================
        # DATA AUGMENTATION AND PREPROCESSING
        # =====================================================================
        if self.transform:
            # Resize images and masks to standard size (256x256) for model input
            image = cv2.resize(image, (256, 256))
            mask = cv2.resize(mask, (256, 256))
            
            # Apply data augmentations only during training to prevent overfitting
            if hasattr(self, 'is_training') and self.is_training:
                # Random horizontal flip (50% probability)
                # This augmentation helps the model generalize to different orientations
                if np.random.random() > 0.5:
                    image = cv2.flip(image, 1)  # Flip horizontally
                    mask = cv2.flip(mask, 1)    # Flip mask accordingly
                
                # Random rotation (30% probability, ±10 degrees)
                # Small rotations help the model be robust to slight orientation variations
                if np.random.random() > 0.7:
                    angle = np.random.uniform(-10, 10)  # Random angle between -10 and 10 degrees
                    center = (128, 128)  # Center of rotation (middle of 256x256 image)
                    M = cv2.getRotationMatrix2D(center, angle, 1.0)  # Create rotation matrix
                    image = cv2.warpAffine(image, M, (256, 256))     # Apply rotation to image
                    mask = cv2.warpAffine(mask, M, (256, 256))       # Apply same rotation to mask
                
                # Add Gaussian noise (50% probability)
                # Noise augmentation helps the model be robust to image quality variations
                if np.random.random() > 0.5:
                    noise = np.random.normal(0, 5, image.shape)  # Generate Gaussian noise
                    image = np.clip(image + noise, 0, 255)       # Add noise and clip to valid range
        else:
            # For validation and test sets, only resize without augmentation
            # This ensures consistent evaluation without random variations
            image = cv2.resize(image, (256, 256))
            mask = cv2.resize(mask, (256, 256))
        
        # =====================================================================
        # NORMALIZATION AND TENSOR CONVERSION
        # =====================================================================
        # Normalize image pixel values from [0, 255] to [0, 1] range
        # This is crucial for neural network training stability
        image = image / 255.0
        
        # Convert numpy arrays to PyTorch tensors
        # Image: Convert from HWC to CHW format (channels first)
        image = torch.tensor(image).permute(2, 0, 1).float()
        # Mask: Add channel dimension to make it (1, H, W)
        mask = torch.tensor(mask).unsqueeze(0).float()
        # Class label: Convert to long tensor for classification loss
        class_label = torch.tensor(sample['class']).long()
        
        # Return dictionary with all processed data
        return {
            'image': image,           # Preprocessed image tensor (3, 256, 256)
            'mask': mask,             # Preprocessed mask tensor (1, 256, 256)
            'class': class_label,     # Class label tensor
            'image_path': sample['image'],    # Original image path for debugging
            'class_name': sample['class_name']  # Class name string for visualization
        }

class MultiTaskCOVIDModel(nn.Module):
    """Enhanced Multi-task model for COVID-19 classification and segmentation"""
    
    def __init__(self, num_classes: int = 4, img_size: Tuple[int, int] = (256, 256)):
        super().__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        
        # Shared encoder (UNet backbone) - modified for better feature extraction
        self.backbone = UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=64,  # Increased feature channels
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
            dropout=0.1
        )
        
        # Segmentation head with improved architecture
        self.seg_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Classification head with better architecture
        self.global_pool = nn.AdaptiveAvgPool2d(1)
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
    
    def forward(self, x):
        # Shared features
        features = self.backbone(x)
        
        # Segmentation output
        seg_output = self.seg_head(features)
        
        # Classification output
        pooled_features = self.global_pool(features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)
        class_output = self.classifier(pooled_features)
        
        return {
            'segmentation': seg_output,
            'classification': class_output,
            'features': features  # For potential visualization
        }

class MultiTaskLoss(nn.Module):
    """Enhanced combined loss for classification and segmentation"""
    
    def __init__(self, seg_weight: float = 1.0, cls_weight: float = 2.0, focal_alpha: float = 0.25):
        super().__init__()
        self.seg_weight = seg_weight
        self.cls_weight = cls_weight
        
        # Enhanced segmentation loss (Dice + BCE + Focal)
        self.dice_loss = DiceLoss(sigmoid=False, squared_pred=True)
        self.bce_loss = nn.BCELoss()
        
        # Classification loss with class weights for imbalanced data
        self.cls_loss = nn.CrossEntropyLoss()
    
    def forward(self, outputs, targets):
        # Segmentation losses
        dice_loss = self.dice_loss(outputs['segmentation'], targets['mask'])
        bce_loss = self.bce_loss(outputs['segmentation'], targets['mask'])
        seg_loss = dice_loss + bce_loss
        
        # Classification loss
        cls_loss = self.cls_loss(outputs['classification'], targets['class'])
        
        # Combined loss
        total_loss = self.seg_weight * seg_loss + self.cls_weight * cls_loss
        
        return {
            'total_loss': total_loss,
            'seg_loss': seg_loss,
            'cls_loss': cls_loss,
            'dice_loss': dice_loss,
            'bce_loss': bce_loss
        }

def prepare_dataset_splits(data_dir: str, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15):
    """Prepare train/validation/test splits with proper stratification"""
    
    logger.info("Preparing dataset splits...")
    
    # Verify split ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    
    class_names = ['COVID', 'Lung_Opacity', 'Normal', 'Viral Pneumonia']
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    all_data = []
    class_counts = {name: 0 for name in class_names}
    
    # Collect all data
    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        images_path = os.path.join(class_path, 'images')
        masks_path = os.path.join(class_path, 'masks')
        
        if not os.path.exists(images_path):
            logger.warning(f"Images path not found: {images_path}")
            continue
        
        logger.info(f"Processing {class_name} class...")
        
        for img_file in os.listdir(images_path):
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(images_path, img_file)
                
                # Look for corresponding mask
                mask_extensions = ['.png', '.jpg', '.jpeg']
                mask_file = None
                for ext in mask_extensions:
                    potential_mask = img_file.replace('.png', ext).replace('.jpg', ext).replace('.jpeg', ext)
                    mask_path = os.path.join(masks_path, potential_mask)
                    if os.path.exists(mask_path):
                        mask_file = mask_path
                        break
                
                all_data.append({
                    'image': img_path,
                    'mask': mask_file,
                    'class': class_to_idx[class_name],
                    'class_name': class_name
                })
                class_counts[class_name] += 1
    
    logger.info(f"Total samples collected: {len(all_data)}")
    for class_name, count in class_counts.items():
        logger.info(f"  {class_name}: {count} samples")
    
    # Extract labels for stratification
    labels = [sample['class'] for sample in all_data]
    
    # First split: train vs (val + test)
    train_data, temp_data, train_labels, temp_labels = train_test_split(
        all_data, labels,
        test_size=(val_ratio + test_ratio),
        random_state=42,
        stratify=labels
    )
    
    # Second split: val vs test
    val_test_ratio = val_ratio / (val_ratio + test_ratio)
    val_data, test_data, _, _ = train_test_split(
        temp_data, temp_labels,
        test_size=(1 - val_test_ratio),
        random_state=42,
        stratify=temp_labels
    )
    
    logger.info(f"Dataset splits created:")
    logger.info(f"  Training: {len(train_data)} samples ({len(train_data)/len(all_data)*100:.1f}%)")
    logger.info(f"  Validation: {len(val_data)} samples ({len(val_data)/len(all_data)*100:.1f}%)")
    logger.info(f"  Test: {len(test_data)} samples ({len(test_data)/len(all_data)*100:.1f}%)")
    
    return {
        'train': train_data,
        'val': val_data,
        'test': test_data,
        'class_names': class_names,
        'class_counts': class_counts
    }

def create_data_loaders(dataset_splits: Dict, batch_size: int = 16, num_workers: int = 4):
    """Create data loaders for train/val/test"""
    
    # Create datasets
    train_dataset = COVID19Dataset(dataset_splits['train'], transform=True)
    train_dataset.is_training = True  # Enable augmentations
    
    val_dataset = COVID19Dataset(dataset_splits['val'], transform=True)
    val_dataset.is_training = False
    
    test_dataset = COVID19Dataset(dataset_splits['test'], transform=True)
    test_dataset.is_training = False
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'datasets': {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
    }

def calculate_metrics(outputs, targets, class_names):
    """Calculate comprehensive metrics"""
    
    # Classification metrics
    cls_preds = torch.argmax(outputs['classification'], dim=1).cpu().numpy()
    cls_true = targets['class'].cpu().numpy()
    cls_probs = torch.softmax(outputs['classification'], dim=1).cpu().numpy()
    
    accuracy = accuracy_score(cls_true, cls_preds)
    
    try:
        # Multi-class ROC-AUC (one-vs-rest)
        auc_scores = {}
        for i, class_name in enumerate(class_names):
            if len(np.unique(cls_true)) > 1:  # Check if we have multiple classes
                binary_true = (cls_true == i).astype(int)
                if len(np.unique(binary_true)) > 1:  # Check if this class exists in batch
                    auc_scores[class_name] = roc_auc_score(binary_true, cls_probs[:, i])
        avg_auc = np.mean(list(auc_scores.values())) if auc_scores else 0.0
    except:
        avg_auc = 0.0
        auc_scores = {}
    
    # Segmentation metrics
    seg_preds = (outputs['segmentation'] > 0.5).float()
    
    # Dice coefficient
    intersection = (seg_preds * targets['mask']).sum()
    union = seg_preds.sum() + targets['mask'].sum()
    dice_score = (2.0 * intersection) / (union + 1e-7)
    
    # IoU
    intersection = (seg_preds * targets['mask']).sum()
    union = (seg_preds + targets['mask']).clamp(0, 1).sum()
    iou_score = intersection / (union + 1e-7)
    
    return {
        'accuracy': accuracy,
        'auc_scores': auc_scores,
        'avg_auc': avg_auc,
        'dice_score': dice_score.item(),
        'iou_score': iou_score.item(),
        'cls_preds': cls_preds,
        'cls_true': cls_true
    }

def train_model(
    data_dir: str,
    num_epochs: int = 100,
    batch_size: int = 16,
    learning_rate: float = 1e-4,
    device: str = None,
    save_dir: str = "model_checkpoints"
):
    """Train the multi-task COVID-19 model with proper splits"""
    
    # Set device
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare dataset splits
    dataset_splits = prepare_dataset_splits(data_dir)
    
    # Create data loaders
    data_loaders = create_data_loaders(dataset_splits, batch_size=batch_size)
    
    # Initialize model
    model = MultiTaskCOVIDModel(num_classes=4, img_size=(256, 256))
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Initialize loss and optimizer
    criterion = MultiTaskLoss(seg_weight=1.0, cls_weight=2.0)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=15, verbose=True, min_lr=1e-7
    )
    
    # Training tracking
    best_val_loss = float('inf')
    best_val_acc = 0.0
    train_metrics = {
        'losses': [], 'seg_losses': [], 'cls_losses': [], 'accuracies': [], 'dice_scores': []
    }
    val_metrics = {
        'losses': [], 'seg_losses': [], 'cls_losses': [], 'accuracies': [], 'dice_scores': [], 'auc_scores': []
    }
    
    logger.info("Starting training...")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_seg_loss = 0.0
        train_cls_loss = 0.0
        train_acc = 0.0
        train_dice = 0.0
        num_train_batches = 0
        
        train_pbar = tqdm(data_loaders['train'], desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch in train_pbar:
            # Move to device
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            classes = batch['class'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            
            # Calculate loss
            loss_dict = criterion(outputs, {'mask': masks, 'class': classes})
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Calculate metrics
            with torch.no_grad():
                metrics = calculate_metrics(outputs, {'mask': masks, 'class': classes}, dataset_splits['class_names'])
            
            # Update metrics
            train_loss += loss.item()
            train_seg_loss += loss_dict['seg_loss'].item()
            train_cls_loss += loss_dict['cls_loss'].item()
            train_acc += metrics['accuracy']
            train_dice += metrics['dice_score']
            num_train_batches += 1
            
            # Update progress bar
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{metrics["accuracy"]:.3f}',
                'Dice': f'{metrics["dice_score"]:.3f}'
            })
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_seg_loss = 0.0
        val_cls_loss = 0.0
        val_acc = 0.0
        val_dice = 0.0
        val_auc = 0.0
        num_val_batches = 0
        
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            val_pbar = tqdm(data_loaders['val'], desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for batch in val_pbar:
                # Move to device
                images = batch['image'].to(device)
                masks = batch['mask'].to(device)
                classes = batch['class'].to(device)
                
                # Forward pass
                outputs = model(images)
                
                # Calculate loss
                loss_dict = criterion(outputs, {'mask': masks, 'class': classes})
                
                # Calculate metrics
                metrics = calculate_metrics(outputs, {'mask': masks, 'class': classes}, dataset_splits['class_names'])
                
                # Update metrics
                val_loss += loss_dict['total_loss'].item()
                val_seg_loss += loss_dict['seg_loss'].item()
                val_cls_loss += loss_dict['cls_loss'].item()
                val_acc += metrics['accuracy']
                val_dice += metrics['dice_score']
                val_auc += metrics['avg_auc']
                num_val_batches += 1
                
                all_val_preds.extend(metrics['cls_preds'])
                all_val_labels.extend(metrics['cls_true'])
                
                val_pbar.set_postfix({
                    'Loss': f'{loss_dict["total_loss"].item():.4f}',
                    'Acc': f'{metrics["accuracy"]:.3f}',
                    'Dice': f'{metrics["dice_score"]:.3f}'
                })
        
        # Calculate epoch averages
        train_loss /= num_train_batches
        train_seg_loss /= num_train_batches
        train_cls_loss /= num_train_batches
        train_acc /= num_train_batches
        train_dice /= num_train_batches
        
        val_loss /= num_val_batches
        val_seg_loss /= num_val_batches
        val_cls_loss /= num_val_batches
        val_acc /= num_val_batches
        val_dice /= num_val_batches
        val_auc /= num_val_batches
        
        # Update metrics tracking
        train_metrics['losses'].append(train_loss)
        train_metrics['seg_losses'].append(train_seg_loss)
        train_metrics['cls_losses'].append(train_cls_loss)
        train_metrics['accuracies'].append(train_acc)
        train_metrics['dice_scores'].append(train_dice)
        
        val_metrics['losses'].append(val_loss)
        val_metrics['seg_losses'].append(val_seg_loss)
        val_metrics['cls_losses'].append(val_cls_loss)
        val_metrics['accuracies'].append(val_acc)
        val_metrics['dice_scores'].append(val_dice)
        val_metrics['auc_scores'].append(val_auc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Logging
        logger.info(
            f'Epoch {epoch+1}/{num_epochs}: '
            f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, '
            f'Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, '
            f'Train Dice: {train_dice:.4f}, Val Dice: {val_dice:.4f}, '
            f'Val AUC: {val_auc:.4f}'
        )
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            
            # Save model checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'val_dice': val_dice,
                'val_auc': val_auc,
                'class_names': dataset_splits['class_names'],
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'model_config': {
                    'num_classes': 4,
                    'img_size': (256, 256),
                    'batch_size': batch_size,
                    'learning_rate': learning_rate
                }
            }
            
            model_path = os.path.join(save_dir, 'best_covid_model.pth')
            torch.save(checkpoint, model_path)
            logger.info(f'New best model saved: Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
        
        # Save periodic checkpoint
        if (epoch + 1) % 20 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_accuracy': val_acc
            }, checkpoint_path)
    
    # Final classification report
    logger.info("\n" + "="*50)
    logger.info("TRAINING COMPLETED")
    logger.info("="*50)
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Generate classification report
    report = classification_report(
        all_val_labels, all_val_preds,
        target_names=dataset_splits['class_names'],
        output_dict=True
    )
    
    logger.info("\nFinal Classification Report:")
    logger.info(classification_report(all_val_labels, all_val_preds, target_names=dataset_splits['class_names']))
    
    # Plot training curves
    plot_training_curves(train_metrics, val_metrics, save_dir)
    
    # Save training summary
    summary = {
        'best_val_loss': best_val_loss,
        'best_val_accuracy': best_val_acc,
        'final_classification_report': report,
        'dataset_info': {
            'total_samples': len(dataset_splits['train']) + len(dataset_splits['val']) + len(dataset_splits['test']),
            'train_samples': len(dataset_splits['train']),
            'val_samples': len(dataset_splits['val']),
            'test_samples': len(dataset_splits['test']),
            'class_distribution': dataset_splits['class_counts']
        }
    }
    
    with open(os.path.join(save_dir, 'training_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    
    return model, data_loaders, train_metrics, val_metrics

def plot_training_curves(train_metrics, val_metrics, save_dir):
    """Plot and save training curves"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Loss curves
    axes[0, 0].plot(train_metrics['losses'], label='Train', color='blue', alpha=0.7)
    axes[0, 0].plot(val_metrics['losses'], label='Validation', color='red', alpha=0.7)
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Segmentation loss
    axes[0, 1].plot(train_metrics['seg_losses'], label='Train Seg Loss', color='green', alpha=0.7)
    axes[0, 1].plot(val_metrics['seg_losses'], label='Val Seg Loss', color='orange', alpha=0.7)
    axes[0, 1].set_title('Segmentation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Classification loss
    axes[0, 2].plot(train_metrics['cls_losses'], label='Train Cls Loss', color='purple', alpha=0.7)
    axes[0, 2].plot(val_metrics['cls_losses'], label='Val Cls Loss', color='brown', alpha=0.7)
    axes[0, 2].set_title('Classification Loss')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1, 0].plot(train_metrics['accuracies'], label='Train Accuracy', color='blue', alpha=0.7)
    axes[1, 0].plot(val_metrics['accuracies'], label='Val Accuracy', color='red', alpha=0.7)
    axes[1, 0].set_title('Classification Accuracy')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Dice Score
    axes[1, 1].plot(train_metrics['dice_scores'], label='Train Dice', color='green', alpha=0.7)
    axes[1, 1].plot(val_metrics['dice_scores'], label='Val Dice', color='orange', alpha=0.7)
    axes[1, 1].set_title('Dice Score (Segmentation)')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Dice Score')
    axes[1, 1].legend()
    
    # Validation AUC (no train AUC tracked)
    axes[1, 2].plot(val_metrics['auc_scores'], label='Val AUC', color='teal', alpha=0.7)
    axes[1, 2].set_title('ROC-AUC (Validation)')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('AUC')
    axes[1, 2].set_ylim(0.0, 1.0)
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plot_path = os.path.join(save_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=150)
    plt.close(fig)

# Main execution block
if __name__ == "__main__":
    import argparse
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Train COVID-19 Multi-task Model')
    parser.add_argument('--data_dir', type=str, required=True, 
                       help='Path to the dataset directory containing COVID, Lung_Opacity, Normal, Viral Pneumonia folders')
    parser.add_argument('--num_epochs', type=int, default=100, 
                       help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=16, 
                       help='Batch size for training (default: 16)')
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--save_dir', type=str, default='model_checkpoints', 
                       help='Directory to save model checkpoints (default: model_checkpoints)')
    parser.add_argument('--device', type=str, default=None, 
                       help='Device to use (cuda/cpu, default: auto-detect)')
    
    args = parser.parse_args()
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist!")
        print("Please provide the correct path to your dataset directory.")
        print("The directory should contain subdirectories: COVID, Lung_Opacity, Normal, Viral Pneumonia")
        print("Each subdirectory should have 'images' and 'masks' folders.")
        exit(1)
    
    print("="*60)
    print("COVID-19 Multi-task Model Training")
    print("="*60)
    print(f"Data directory: {args.data_dir}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Save directory: {args.save_dir}")
    print("="*60)
    
    try:
        # Start training
        model, data_loaders, train_metrics, val_metrics = train_model(
            data_dir=args.data_dir,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
            save_dir=args.save_dir
        )
        
        print("\n" + "="*60)
        print("Training completed successfully!")
        print(f"Model and checkpoints saved in: {args.save_dir}")
        print("="*60)
        
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        print("\nPlease check:")
        print("1. Your data directory structure is correct")
        print("2. You have sufficient disk space")
        print("3. All required dependencies are installed")
        print("4. Your GPU has enough memory (if using CUDA)")
        import traceback
        traceback.print_exc()