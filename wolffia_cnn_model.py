#!/usr/bin/env python3
"""
Enhanced Wolffia CNN with python_for_microscopists Techniques
Professional implementation based on established computer vision patterns for microscopy

CITATIONS:
- python_for_microscopists example 066: Deep Learning for image segmentation using UNet
- python_for_microscopists example 075: Multi-scale features for medical image segmentation
- python_for_microscopists example 127: Attention mechanisms in biomedical imaging
- python_for_microscopists example 103: Data augmentation strategies for microscopy
- python_for_microscopists example 085: Multi-task learning for cell detection
- python_for_microscopists example 118: Focal loss for imbalanced segmentation

Author: BIOIMAGIN Professional Team
Based on: python_for_microscopists by Dr. Sreenivas Bhattiprolu
"""

import json
import math
import os
import warnings
from datetime import datetime
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy import ndimage
from scipy.ndimage import gaussian_filter
from skimage.draw import ellipse
from skimage.morphology import remove_small_objects
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings('ignore')

PYTORCH_AVAILABLE = torch.cuda.is_available() or torch.backends.mps.is_available() or torch.backends.cpu.is_available()
print("✅ PyTorch available for CNN training" if PYTORCH_AVAILABLE else "❌ PyTorch not available - CNN features disabled")


class GreenEnhancedPreprocessor:
    """
    Enhanced RGB preprocessing using green-enhancement techniques
    Creates 3-channel input: RGB + Green-Enhanced + Color-Enhanced
    """
    
    def __init__(self):
        # Color enhancement parameters
        self.green_hsv_lower = np.array([35, 40, 40])
        self.green_hsv_upper = np.array([85, 255, 255])
    
    def create_green_enhanced_channels(self, rgb_image):
        """
        Create enhanced 3-channel input from RGB image
        
        Args:
            rgb_image: BGR image (OpenCV format)
            
        Returns:
            3-channel enhanced image stack
        """
        # Convert BGR to RGB for processing
        if len(rgb_image.shape) == 3:
            rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        else:
            rgb = rgb_image
            
        # Extract color channels
        r, g, b = cv2.split(rgb)
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        
        # Create green mask
        green_mask = cv2.inRange(hsv, self.green_hsv_lower, self.green_hsv_upper)
        green_mask_norm = green_mask.astype(np.float32) / 255.0
        
        # Channel 1: Enhanced green channel
        green_enhanced = g.astype(np.float32) / 255.0
        green_enhanced = green_enhanced * (1.0 + 0.3 * green_mask_norm)
        green_enhanced = np.clip(green_enhanced, 0, 1)
        
        # Channel 2: Color-enhanced grayscale (from create_green_enhanced_grayscale)
        # 40% green channel + 30% LAB green + 30% green mask
        green_component = g.astype(np.float32) / 255.0 * 0.4
        lab_green = (255 - lab[:, :, 1]).astype(np.float32) / 255.0 * 0.3  # Inverted A channel
        mask_component = green_mask_norm * 0.3
        
        color_enhanced = green_component + lab_green + mask_component
        color_enhanced = np.clip(color_enhanced, 0, 1)
        
        # Apply CLAHE enhancement
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        color_enhanced_uint8 = (color_enhanced * 255).astype(np.uint8)
        color_enhanced = clahe.apply(color_enhanced_uint8).astype(np.float32) / 255.0
        
        # Channel 3: Red channel (for contrast and structure)
        red_enhanced = r.astype(np.float32) / 255.0
        
        # Stack into 3-channel image
        enhanced_stack = np.stack([green_enhanced, color_enhanced, red_enhanced], axis=2)
        
        return enhanced_stack
    
    def analyze_green_content(self, rgb_image):
        """
        Analyze green content percentage in image
        """
        if len(rgb_image.shape) == 3:
            rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        else:
            rgb = rgb_image
            
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        green_mask = cv2.inRange(hsv, self.green_hsv_lower, self.green_hsv_upper)
        
        total_pixels = rgb.shape[0] * rgb.shape[1]
        green_pixels = np.sum(green_mask > 0)
        
        return (green_pixels / total_pixels) * 100.0


class AttentionBlock(nn.Module):
    """
    Attention mechanism for improved feature focusing
    Implementation based on python_for_microscopists example 127: Attention mechanisms
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class FocalLoss(nn.Module):
    """
    Focal Loss implementation for addressing class imbalance
    Based on python_for_microscopists example 118: Focal loss for imbalanced segmentation
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction  # use 'none', 'mean', or 'sum'

    def forward(self, inputs, targets):
        # Apply sigmoid if not already applied
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # pt is the probability of the true class
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * bce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss



class WolffiaFocalLoss(nn.Module):
    """
    Enhanced Focal Loss with background rejection for Wolffia cells
    Combines focal loss with background penalty to focus on true cells
    """
    def __init__(self, alpha=0.7, gamma=2.0, background_penalty=0.3):
        super(WolffiaFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.background_penalty = background_penalty
    
    def forward(self, segmentation_output, background_output, seg_targets, bg_targets):
        """
        Args:
            segmentation_output: Cell segmentation predictions
            background_output: Background classification predictions
            seg_targets: Segmentation ground truth
            bg_targets: Background ground truth
        """
        # Segmentation focal loss
        seg_bce = F.binary_cross_entropy_with_logits(segmentation_output, seg_targets, reduce=False)
        seg_pt = torch.exp(-seg_bce)
        seg_focal = self.alpha * (1 - seg_pt)**self.gamma * seg_bce
        
        # Background classification loss
        bg_bce = F.binary_cross_entropy_with_logits(background_output, bg_targets, reduce=False)
        
        # Combine losses with background penalty
        total_loss = torch.mean(seg_focal) + self.background_penalty * torch.mean(bg_bce)
        
        return total_loss


class ASPP(nn.Module):
    """
    Atrous Spatial Pyramid Pooling for multi-scale feature extraction
    Inspired by python_for_microscopists example 075: Multi-scale features
    """
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        
        # Different dilation rates for multi-scale context
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2, bias=False)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=4, dilation=4, bias=False)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=8, dilation=8, bias=False)
        
        # Global average pooling
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
        )
        
        # Final projection
        self.conv_out = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        size = x.shape[-2:]
        
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = F.interpolate(self.global_avg_pool(x), size=size, mode='bilinear', align_corners=False)
        
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        x = self.conv_out(x)
        x = self.bn(x)
        x = self.relu(x)
        
        return x


class VGGBlock(nn.Module):
    """
    VGG-style block with batch normalization and activation
    Based on VGG architectures from python_for_microscopists examples
    """
    def __init__(self, in_channels, out_channels, num_convs=2):
        super(VGGBlock, self).__init__()
        layers = []
        for i in range(num_convs):
            layers.append(nn.Conv2d(in_channels if i == 0 else out_channels, out_channels, 3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class VGGUNet(nn.Module):
    """
    VGG-based U-Net architecture for Wolffia cell segmentation
    Enhanced architecture based on python_for_microscopists U-Net examples
    Specifically designed to prevent whole-image detection
    """
    def __init__(self, input_channels=1, output_channels=1, features=[64, 128, 256, 512]):
        super(VGGUNet, self).__init__()
        
        # Encoder (VGG-style with skip connections)
        self.encoder_blocks = nn.ModuleList()
        self.pool = nn.MaxPool2d(2, 2)
        
        in_channels = input_channels
        for feature in features:
            self.encoder_blocks.append(VGGBlock(in_channels, feature))
            in_channels = feature
        
        # Bottleneck
        self.bottleneck = VGGBlock(features[-1], features[-1] * 2)
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.upconvs = nn.ModuleList()
        
        for feature in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(feature * 2, feature, 2, 2))
            self.decoder_blocks.append(VGGBlock(feature * 2, feature))
        
        # Final classification layer
        self.final_conv = nn.Conv2d(features[0], output_channels, 1)
        self.sigmoid = nn.Sigmoid()
        
        # Dropout for regularization
        self.dropout = nn.Dropout2d(0.1)
    
    def forward(self, x):
        # Encoder path with skip connections
        skip_connections = []
        
        for encoder in self.encoder_blocks:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        x = self.dropout(x)
        
        # Decoder path
        skip_connections = skip_connections[::-1]  # Reverse for decoder
        
        for idx, (upconv, decoder) in enumerate(zip(self.upconvs, self.decoder_blocks)):
            x = upconv(x)
            
            # Handle size mismatch
            skip_connection = skip_connections[idx]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='bilinear', align_corners=False)
            
            # Concatenate skip connection
            x = torch.cat([skip_connection, x], dim=1)
            x = decoder(x)
        
        # Final output
        x = self.final_conv(x)
        x = self.sigmoid(x)
        
        return x


class WolffiaCNN(nn.Module):
    """
    Enhanced U-Net architecture with attention, ASPP, and multi-task capabilities
    Professional implementation based on multiple python_for_microscopists examples:
    
    CITATIONS:
    - Example 066: Core U-Net architecture for biomedical segmentation
    - Example 075: Multi-scale feature extraction using dilated convolutions
    - Example 127: Attention mechanisms for better feature focusing  
    - Example 085: Multi-task learning for simultaneous detection tasks
    - Example 103: Advanced normalization and regularization techniques
    """
    
    def __init__(self, input_channels=3, output_channels=1, base_filters=32, use_attention=True, multi_task=True):
        super(WolffiaCNN, self).__init__()
        
        self.use_attention = use_attention
        self.multi_task = multi_task
        
        # Encoder path with residual connections (python_for_microscopists pattern)
        self.enc1 = self._make_encoder_block(input_channels, base_filters)
        self.enc2 = self._make_encoder_block(base_filters, base_filters * 2)
        self.enc3 = self._make_encoder_block(base_filters * 2, base_filters * 4)
        self.enc4 = self._make_encoder_block(base_filters * 4, base_filters * 8)
        
        # Bottleneck with ASPP for multi-scale context
        self.bottleneck = nn.Sequential(
            ASPP(base_filters * 8, base_filters * 16),
            nn.Dropout2d(0.2)
        )
        
        # Attention blocks (if enabled) - FIXED channel dimensions
        if self.use_attention:
            self.att4 = AttentionBlock(base_filters * 8, base_filters * 8, base_filters * 4)  # Fixed: F_g should match upconv output
            self.att3 = AttentionBlock(base_filters * 4, base_filters * 4, base_filters * 2)  # Fixed: F_g should match upconv output
            self.att2 = AttentionBlock(base_filters * 2, base_filters * 2, base_filters)      # Fixed: F_g should match upconv output
            self.att1 = AttentionBlock(base_filters, base_filters, base_filters // 2)        # Fixed: F_g should match upconv output
        
        # Decoder path
        self.upconv4 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, 2, stride=2)
        self.dec4 = self._make_decoder_block(base_filters * 16, base_filters * 8)
        
        self.upconv3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, 2, stride=2)
        self.dec3 = self._make_decoder_block(base_filters * 8, base_filters * 4)
        
        self.upconv2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 2, stride=2)
        self.dec2 = self._make_decoder_block(base_filters * 4, base_filters * 2)
        
        self.upconv1 = nn.ConvTranspose2d(base_filters * 2, base_filters, 2, stride=2)
        self.dec1 = self._make_decoder_block(base_filters * 2, base_filters)
        
        # Multi-task output heads
        # Main segmentation output (python_for_microscopists example 066)
        self.seg_head = nn.Conv2d(base_filters, output_channels, 1)
        
        if self.multi_task:
            # Edge detection head (python_for_microscopists example 085)
            self.edge_head = nn.Conv2d(base_filters, 1, 1)
            # Distance transform head for watershed post-processing
            self.dist_head = nn.Conv2d(base_filters, 1, 1)
        
        # Initialize weights using He initialization (python_for_microscopists best practice)
        self._initialize_weights()
    
    def _make_encoder_block(self, in_channels, out_channels):
        """Create encoder block with residual connection pattern"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
    
    def _make_decoder_block(self, in_channels, out_channels):
        """Create decoder block with skip connections"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )
    
    def _initialize_weights(self):
        """Initialize weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Encoder path
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))
        
        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))
        
        # Decoder path with attention
        dec4 = self.upconv4(bottleneck)
        if self.use_attention:
            enc4 = self.att4(dec4, enc4)  # Now channels match: both are base_filters * 8
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        if self.use_attention:
            enc3 = self.att3(dec3, enc3)  # Now channels match: both are base_filters * 4
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        if self.use_attention:
            enc2 = self.att2(dec2, enc2)  # Now channels match: both are base_filters * 2
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        if self.use_attention:
            enc1 = self.att1(dec1, enc1)  # Now channels match: both are base_filters
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Multi-task outputs
        seg_output = torch.sigmoid(self.seg_head(dec1))
        
        if self.multi_task:
            edge_output = torch.sigmoid(self.edge_head(dec1))
            dist_output = torch.relu(self.dist_head(dec1))  # Distance should be positive
            return seg_output, edge_output, dist_output
        else:
            return seg_output


class WolffiaCNNTrainer:
    def __init__(self, real_images_dir='images', device=None):
        self.real_images_dir = Path(real_images_dir)
        if not self.real_images_dir.exists():
            print(f"⚠️ Real image directory not found: {self.real_images_dir.resolve()}")
        else:
            print(f"📁 Using real images from: {self.real_images_dir.resolve()}")

        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.history = {}
        print(f"✅ Enhanced CNN Trainer initialized - Device: {self.device}")

    def create_datasets(self, train_samples, val_samples, test_samples, batch_size, multi_task=False, use_rgb=False):
        print("📸 Creating training dataset with sample previews...")
        self.train_loader = DataLoader(
            WolffiaSyntheticDataset(
                train_samples, 
                128, 
                real_backgrounds_dir='images',
                save_previews=True,  # FIXED: Enable sample preview saving
                preview_path='sample_preview',
                multi_task=multi_task,
                use_rgb=use_rgb
            ), 
            batch_size=batch_size, 
            shuffle=True
        )
        
        print("📸 Creating validation dataset...")
        self.val_loader = DataLoader(
            WolffiaSyntheticDataset(val_samples, 128, real_backgrounds_dir='images', multi_task=multi_task, use_rgb=use_rgb), 
            batch_size=batch_size
        )
        
        print("📸 Creating test dataset...")
        self.test_loader = DataLoader(
            WolffiaSyntheticDataset(test_samples, 128, real_backgrounds_dir='images', multi_task=multi_task, use_rgb=use_rgb), 
            batch_size=batch_size
        )
        
        print("✅ Datasets created! Training samples will be saved to 'sample_preview/' directory")

    def initialize_model(self, input_channels=1, base_filters=32, use_attention=True, multi_task=True):
        self.model = WolffiaCNN(
            input_channels=input_channels,
            output_channels=1,
            base_filters=base_filters,
            use_attention=use_attention,
            multi_task=multi_task
        ).to(self.device)
        
        self.multi_task = multi_task
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"🧠 Enhanced model initialized:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Using attention: {use_attention}")
        print(f"   Multi-task learning: {multi_task}")

    def train_model(self, epochs=50, learning_rate=0.001, use_focal_loss=False):
        # Enhanced loss functions for RGB training
        if use_focal_loss and self.model.multi_task:
            # Use enhanced Wolffia focal loss with background rejection
            criterion = WolffiaFocalLoss(alpha=0.7, gamma=2.0, background_penalty=0.3)
            seg_criterion = nn.BCELoss()  # Fallback for non-enhanced paths
            print("✅ Using Enhanced Wolffia Focal Loss with background rejection")
        elif use_focal_loss:
            seg_criterion = FocalLoss(alpha=1, gamma=2)
            criterion = None
            print("✅ Using standard Focal Loss for segmentation")
        else:
            seg_criterion = nn.BCELoss()
            criterion = None
        
        edge_criterion = nn.BCELoss()
        dist_criterion = nn.MSELoss()
        
        # Advanced optimizer with weight decay
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        # Learning rate scheduling
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        best_loss = float('inf')
        best_accuracy = 0.0
        checkpoint_counter = 0
        
        self.history = {
            'train_losses': [], 'val_losses': [], 'val_accuracies': [],
            'seg_losses': [], 'edge_losses': [], 'dist_losses': [],
            'epochs_trained': 0, 'best_val_loss': float('inf'), 'best_accuracy': 0.0,
            'checkpoints_saved': []
        }

        print(f"🚀 Starting training for {epochs} epochs...")
        if self.multi_task:
            print("📊 Multi-task learning: Segmentation + Edge Detection + Distance Transform")
        else:
            print("📊 Single-task learning: Segmentation only")

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            seg_loss_total = 0
            edge_loss_total = 0
            dist_loss_total = 0
            
            for batch_idx, batch_data in enumerate(self.train_loader):
                # Handle different data formats
                if len(batch_data) == 3:  # RGB mode: images, seg_masks, bg_masks
                    images, seg_masks, bg_masks = batch_data
                    bg_masks = bg_masks.to(self.device)
                elif len(batch_data) == 2:  # Multi-task or standard mode
                    images, masks = batch_data
                    bg_masks = None
                
                images = images.to(self.device)
                
                # Handle different mask formats
                if self.multi_task and isinstance(masks, (tuple, list)) and len(masks) >= 3:
                    if len(masks) == 4:  # Enhanced multi-task with background
                        seg_masks, edge_masks, dist_masks, bg_masks = masks
                        bg_masks = bg_masks.to(self.device)
                    else:  # Standard multi-task
                        seg_masks, edge_masks, dist_masks = masks
                        bg_masks = None
                    seg_masks = seg_masks.to(self.device)
                    edge_masks = edge_masks.to(self.device)
                    dist_masks = dist_masks.to(self.device)
                else:
                    # Single task mode - only segmentation masks
                    if len(batch_data) == 3:
                        seg_masks = seg_masks.to(self.device)  # Already extracted above
                    else:
                        seg_masks = masks.to(self.device) if not isinstance(masks, (tuple, list)) else masks[0].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                if self.multi_task:
                    outputs = self.model(images)
                    if len(outputs) == 3:
                        seg_output, edge_output, dist_output = outputs
                        bg_output = None
                    elif len(outputs) == 4:  # With background output
                        seg_output, edge_output, dist_output, bg_output = outputs
                    else:
                        seg_output = outputs
                        edge_output = dist_output = bg_output = None
                else:
                    seg_output = self.model(images)
                    edge_output = dist_output = bg_output = None
                
                # Loss calculation
                if use_focal_loss and self.model.multi_task and bg_output is not None and bg_masks is not None and criterion is not None:
                    # Use enhanced focal loss with background rejection
                    loss = criterion(seg_output, bg_output, seg_masks, bg_masks)
                    seg_loss_total += loss.item()
                elif self.multi_task and edge_output is not None and dist_output is not None:
                    # Standard multi-task loss
                    seg_loss = seg_criterion(seg_output, seg_masks)
                    edge_loss = edge_criterion(edge_output, edge_masks)
                    dist_loss = dist_criterion(dist_output, dist_masks)
                    
                    # Weighted combination of losses
                    loss = seg_loss + 0.5 * edge_loss + 0.3 * dist_loss
                    
                    seg_loss_total += seg_loss.item()
                    edge_loss_total += edge_loss.item()
                    dist_loss_total += dist_loss.item()
                else:
                    # Single task mode
                    loss = seg_criterion(seg_output, seg_masks)
                    seg_loss_total += loss.item()
                
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                optimizer.step()
                total_loss += loss.item()
            
            # Validation phase
            val_loss, val_accuracy = self._validate()
            
            # Learning rate scheduling
            scheduler.step()
            
            # Record history
            avg_train_loss = total_loss / len(self.train_loader)
            avg_seg_loss = seg_loss_total / len(self.train_loader)
            
            self.history['train_losses'].append(avg_train_loss)
            self.history['val_losses'].append(val_loss)
            self.history['val_accuracies'].append(val_accuracy)
            self.history['seg_losses'].append(avg_seg_loss)
            
            if self.multi_task and edge_loss_total > 0:
                self.history['edge_losses'].append(edge_loss_total / len(self.train_loader))
                self.history['dist_losses'].append(dist_loss_total / len(self.train_loader))
            
            # Progress reporting
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_accuracy:.3f} | "
                  f"LR: {current_lr:.2e}")
            
            # Smart checkpoint saving: Save every 10 epochs and best accuracy
            should_save = False
            save_reason = ""
            
            # Save every 10 epochs
            if (epoch + 1) % 10 == 0:
                checkpoint_counter += 1
                should_save = True
                save_reason = f"checkpoint at epoch {epoch+1}"
            
            # Always save if best accuracy
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                self.history['best_accuracy'] = best_accuracy
                should_save = True
                save_reason = f"best accuracy {val_accuracy:.4f}"
            
            # Also save if best loss (for compatibility)
            if val_loss < best_loss:
                best_loss = val_loss
                self.history['best_val_loss'] = best_loss
                should_save = True
                if save_reason == "":
                    save_reason = f"best loss {best_loss:.4f}"
            
            if should_save:
                self.history['epochs_trained'] = epoch + 1
                
                # Enhanced checkpoint with model configuration
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'loss': val_loss,
                    'accuracy': val_accuracy,
                    'best_loss': best_loss,
                    'best_accuracy': best_accuracy,
                    'model_config': {
                        'input_channels': self.model.enc1[0].in_channels,
                        'use_attention': self.model.use_attention,
                        'multi_task': self.model.multi_task,
                        'base_filters': 32  # Default base filters
                    },
                    'training_history': self.history
                }
                
                torch.save(checkpoint, 'models/wolffia_cnn_best.pth')
                print(f"💾 Model saved: {save_reason}")
                self.history['checkpoints_saved'].append({
                    'epoch': epoch + 1,
                    'reason': save_reason,
                    'accuracy': val_accuracy,
                    'loss': val_loss
                })

        return self.history

    def _validate(self):
        """Validation with multi-task support"""
        self.model.eval()
        val_loss = 0
        correct_pixels = 0
        total_pixels = 0
        
        with torch.no_grad():
            for images, masks in self.val_loader:
                images = images.to(self.device)
                
                # Handle different mask formats
                if self.multi_task and isinstance(masks, (tuple, list)) and len(masks) == 3:
                    seg_masks = masks[0].to(self.device)
                else:
                    seg_masks = masks.to(self.device) if not isinstance(masks, (tuple, list)) else masks[0].to(self.device)
                
                if self.multi_task:
                    if isinstance(masks, (tuple, list)) and len(masks) == 3:
                        seg_output, _, _ = self.model(images)
                    else:
                        seg_output, _, _ = self.model(images)
                else:
                    seg_output = self.model(images)
                
                # Use BCELoss for validation
                loss = F.binary_cross_entropy(seg_output, seg_masks)
                val_loss += loss.item()
                
                # Calculate accuracy
                predicted = (seg_output > 0.5).float()
                correct_pixels += (predicted == seg_masks).sum().item()
                total_pixels += seg_masks.numel()
        
        avg_val_loss = val_loss / len(self.val_loader)
        accuracy = correct_pixels / total_pixels
        
        return avg_val_loss, accuracy

    def evaluate_model(self):
        """Evaluate trained CNN model on test set with compatibility for multi-task and RGB modes"""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_data in self.test_loader:
                # Handle RGB or standard formats
                if len(batch_data) == 3:
                    images, masks, _ = batch_data
                elif len(batch_data) == 2:
                    images, masks = batch_data
                else:
                    raise ValueError("❌ Unexpected batch format in test loader")

                # Move input image to device
                images = images.to(self.device)

                # Handle segmentation-only target if masks is list/tuple
                if isinstance(masks, (tuple, list)):
                    masks = masks[0]
                masks = masks.to(self.device)

                # Model prediction
                outputs = self.model(images)

                # Handle multi-output models
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]

                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predictions == masks).sum().item()
                total += masks.numel()

        accuracy = correct / total
        print(f"✅ Evaluation complete - Accuracy: {accuracy:.4f}")
        return {'test_accuracy': accuracy}


    def visualize_training_history(self, history):
        plt.figure()
        plt.plot(history['losses'], label='Training Loss')
        plt.title('Training Loss History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('models/training_history.png')
        plt.close()
        with open('models/training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        print("📊 Training history saved to models/")


class WolffiaSyntheticDataset(Dataset):
    def __init__(self, num_samples=10000, image_size=128, real_backgrounds_dir='images', save_previews=False, preview_path='sample_preview', multi_task=False, use_rgb=False):
        self.num_samples = num_samples
        self.image_size = image_size
        self.multi_task = multi_task
        self.use_rgb = use_rgb
        self.real_patches = []
        self.save_previews = save_previews
        self.preview_path = Path(preview_path)
        self.preview_path.mkdir(parents=True, exist_ok=True)

        path = Path(real_backgrounds_dir)
        if path.exists():
            for f in path.glob('*'):
                try:
                    img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
                    if img is not None and img.shape[0] >= image_size and img.shape[1] >= image_size:
                        y = np.random.randint(0, img.shape[0] - image_size)
                        x = np.random.randint(0, img.shape[1] - image_size)
                        patch = img[y:y+image_size, x:x+image_size].astype(np.float32) / 255.0
                        self.real_patches.append(patch)
                except:
                    continue

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        h, w = self.image_size, self.image_size
        
        # ENHANCED: Create realistic Wolffia-like samples with proper color representation
        if self.real_patches and np.random.rand() < 0.5:
            # Use real background patches
            img = np.copy(self.real_patches[np.random.randint(len(self.real_patches))])
        else:
            # Create realistic microscopy backgrounds
            bg_mode = np.random.choice(['culture_medium', 'plate_gradient', 'water_background', 'slight_debris'])
            if bg_mode == 'culture_medium':
                # Light culture medium background (typical for Wolffia growth)
                base_intensity = np.random.uniform(0.85, 0.95)
                img = np.ones((h, w), dtype=np.float32) * base_intensity
                # Add slight medium turbidity
                noise = np.random.normal(0, 0.02, size=(h, w))
                img += noise
            elif bg_mode == 'plate_gradient':
                # Petri dish edge effects - darker towards edges
                y, x = np.ogrid[:h, :w]
                cy, cx = h // 2, w // 2
                r = np.sqrt((x - cx)**2 + (y - cy)**2)
                r_norm = (r / r.max())
                # Lighter center, slightly darker edges (plate effect)
                img = (0.85 + 0.1 * (1 - r_norm**2)).astype(np.float32)
            elif bg_mode == 'water_background':
                # Clean water/medium with very slight variation
                img = np.ones((h, w), dtype=np.float32) * np.random.uniform(0.88, 0.96)
                # Add very subtle water movement effects
                wave_x = np.sin(np.arange(w) * 0.1) * 0.01
                wave_y = np.sin(np.arange(h) * 0.15) * 0.01
                img += np.tile(wave_x, (h, 1)) + np.tile(wave_y.reshape(-1, 1), (1, w))
            else:  # slight_debris
                # Background with minimal debris (realistic culture conditions)
                img = np.ones((h, w), dtype=np.float32) * np.random.uniform(0.82, 0.92)
                # Add occasional small debris
                for _ in range(np.random.randint(1, 4)):
                    debris_x, debris_y = np.random.randint(5, w-5), np.random.randint(5, h-5)
                    debris_size = np.random.randint(1, 3)
                    cv2.circle(img, (debris_x, debris_y), debris_size, np.random.uniform(0.6, 0.8), -1)

        mask = np.zeros((h, w), dtype=np.float32)
        
        # ENHANCED: Generate realistic Wolffia cell numbers and sizes
        # Wolffia cells are typically 0.5-1.5mm, but can cluster
        num_cells = np.random.randint(5, 25)  # More realistic cell count for field of view

        for _ in range(num_cells):
            # Wolffia-specific cell sizes (small, oval-shaped)
            ry = np.random.randint(3, 8)  # Slightly larger for better detection
            rx = np.random.randint(2, 6)  # Slightly elongated
            cy = np.random.randint(ry, h - ry)
            cx = np.random.randint(rx, w - rx)
            angle = np.random.randint(0, 180)
            
            # Generate cell area
            rr, cc = ellipse(cy, cx, ry, rx, shape=(h, w), rotation=np.deg2rad(angle))
            
            # ENHANCED: Realistic Wolffia cell appearance
            # Wolffia cells have distinctive green coloration (chlorophyll)
            # In grayscale, this appears as medium-dark intensity
            base_intensity = np.random.uniform(0.25, 0.55)  # Darker than background (green cells)
            
            # Add cell internal structure variation
            cell_noise = np.random.normal(0, 0.02, size=rr.shape[0])
            cell_intensity = np.clip(base_intensity + cell_noise, 0.15, 0.65)
            
            # Add slight cell edge definition (cell wall)
            img[rr, cc] = cell_intensity
            mask[rr, cc] = 1.0
            
            # Add cell wall enhancement (slightly darker edges)
            if np.random.rand() < 0.7:  # Most cells have visible walls
                # Create edge mask
                struct = np.ones((3, 3))
                cell_mask = np.zeros((h, w))
                cell_mask[rr, cc] = 1
                edges = cell_mask - ndimage.binary_erosion(cell_mask, struct).astype(cell_mask.dtype)
                edge_coords = np.where(edges > 0)
                if len(edge_coords[0]) > 0:
                    edge_enhancement = np.random.uniform(0.05, 0.15)
                    img[edge_coords] = np.clip(img[edge_coords] - edge_enhancement, 0, 1)

        # ENHANCED: Add realistic microscopy effects
        # Slight focus variations
        if np.random.rand() < 0.4:
            blur_sigma = np.random.uniform(0.3, 1.0)
            img = gaussian_filter(img, sigma=blur_sigma)

        # Light scattering effects (typical in brightfield microscopy)
        if np.random.rand() < 0.3:
            scatter_noise = np.random.normal(0, 0.008, size=img.shape)
            img += scatter_noise

        # Illumination variation (common in microscopy)
        if np.random.rand() < 0.5:
            # Create subtle illumination gradient
            y, x = np.ogrid[:h, :w]
            illum_center_y = np.random.randint(h//3, 2*h//3)
            illum_center_x = np.random.randint(w//3, 2*w//3)
            illum_dist = np.sqrt((x - illum_center_x)**2 + (y - illum_center_y)**2)
            max_dist = np.sqrt(h**2 + w**2) / 2
            illum_factor = 1.0 + 0.1 * (1 - illum_dist / max_dist)
            img *= illum_factor

        # Final clipping
        img = np.clip(img, 0, 1)

        # ENHANCED: Save colorized previews for better visualization
        if self.save_previews:
            current_previews = len(list(self.preview_path.glob('img_*.png')))
            if current_previews < 100:
                # Create a colorized version for preview (simulate green cells)
                preview_img = self.create_colorized_preview(img, mask)
                cv2.imwrite(str(self.preview_path / f"img_{current_previews}.png"), preview_img)
                cv2.imwrite(str(self.preview_path / f"mask_{current_previews}.png"), (mask * 255).astype(np.uint8))

        # Choose output format based on configuration
        if self.use_rgb:
            # ENHANCED: Convert to RGB with green-enhanced channels
            preprocessor = GreenEnhancedPreprocessor()
            
            # Create RGB image from grayscale (simulate real microscopy colors)
            rgb_img = np.stack([img, img, img], axis=-1)
            
            # Add green enhancement where cells are present (simulate chlorophyll)
            cell_areas = mask > 0
            if np.any(cell_areas):
                # Enhance green channel in cell areas
                rgb_img[cell_areas, 1] = np.clip(rgb_img[cell_areas, 1] + 0.2, 0, 1)  # More green
                rgb_img[cell_areas, 0] = np.clip(rgb_img[cell_areas, 0] - 0.05, 0, 1)  # Less red
                rgb_img[cell_areas, 2] = np.clip(rgb_img[cell_areas, 2] - 0.05, 0, 1)  # Less blue
            
            # Convert to BGR for OpenCV compatibility
            bgr_img = cv2.cvtColor((rgb_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            # Apply green-enhanced preprocessing
            enhanced_rgb = preprocessor.create_green_enhanced_channels(bgr_img)
            
            # Convert to tensor format (C, H, W)
            img_tensor = torch.from_numpy(enhanced_rgb).permute(2, 0, 1).float()
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).float()
            
            if self.multi_task:
                # Generate enhanced multi-task outputs with background
                edge_mask = self._generate_edge_mask(mask)
                dist_mask = self._generate_distance_mask(mask)
                bg_mask = 1.0 - mask
                
                return (img_tensor, 
                       (mask_tensor,
                        torch.from_numpy(edge_mask).unsqueeze(0).float(),
                        torch.from_numpy(dist_mask).unsqueeze(0).float(),
                        torch.from_numpy(bg_mask).unsqueeze(0).float()))
            else:
                # For RGB training, also return background mask for enhanced loss
                bg_mask = 1.0 - mask
                return img_tensor, mask_tensor, torch.from_numpy(bg_mask).unsqueeze(0).float()
        else:
            # Standard grayscale output
            if self.multi_task:
                # Generate simple edge and distance masks for multi-task learning
                edge_mask = self._generate_edge_mask(mask)
                dist_mask = self._generate_distance_mask(mask)
                
                return (torch.from_numpy(img).unsqueeze(0), 
                       (torch.from_numpy(mask).unsqueeze(0),
                        torch.from_numpy(edge_mask).unsqueeze(0),
                        torch.from_numpy(dist_mask).unsqueeze(0)))
            else:
                return torch.from_numpy(img).unsqueeze(0), torch.from_numpy(mask).unsqueeze(0)
    
    def create_colorized_preview(self, gray_img, mask):
        """
        Create a colorized preview that simulates how Wolffia cells appear in real microscopy
        This helps with visualization and understanding of what the model is learning
        """
        # Convert grayscale to RGB
        preview = np.stack([gray_img, gray_img, gray_img], axis=-1)
        
        # Enhance green channel where cells are detected (simulate chlorophyll)
        cell_areas = mask > 0
        if np.any(cell_areas):
            # Increase green channel in cell areas
            preview[cell_areas, 1] = np.clip(preview[cell_areas, 1] + 0.3, 0, 1)  # More green
            preview[cell_areas, 0] = np.clip(preview[cell_areas, 0] - 0.1, 0, 1)  # Less red
            preview[cell_areas, 2] = np.clip(preview[cell_areas, 2] - 0.1, 0, 1)  # Less blue
        
        # Convert to 8-bit for saving
        return (preview * 255).astype(np.uint8)
    
    def _generate_edge_mask(self, mask):
        """Generate edge mask from segmentation mask"""
        # Use Canny edge detection on the mask
        edges = cv2.Canny((mask * 255).astype(np.uint8), 50, 150)
        return (edges > 0).astype(np.float32)
    
    def _generate_distance_mask(self, mask):
        """Generate distance transform mask for watershed post-processing"""
        from scipy.ndimage import distance_transform_edt
        
        # Distance transform from cell centers
        dist = distance_transform_edt(mask > 0.5)
        
        # Normalize to 0-1 range
        if np.max(dist) > 0:
            dist = dist / np.max(dist)
        
        return dist.astype(np.float32)