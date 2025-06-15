import json
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

warnings.filterwarnings('ignore')

PYTORCH_AVAILABLE = torch.cuda.is_available() or torch.backends.mps.is_available() or torch.backends.cpu.is_available()
print("✅ PyTorch available for CNN training" if PYTORCH_AVAILABLE else "❌ PyTorch not available - CNN features disabled")


warnings.filterwarnings('ignore')

PYTORCH_AVAILABLE = torch.cuda.is_available() or torch.backends.mps.is_available() or torch.backends.cpu.is_available()
print("✅ PyTorch available for CNN training" if PYTORCH_AVAILABLE else "❌ PyTorch not available - CNN features disabled")


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
    Original lightweight CNN - kept for backward compatibility
    """
    def __init__(self, input_channels=1, output_channels=1):
        super(WolffiaCNN, self).__init__()
        self.enc1 = self.double_conv(input_channels, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self.double_conv(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = self.double_conv(32, 64)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = self.double_conv(64, 128)
        self.upconv3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = self.double_conv(128, 64)
        self.upconv2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = self.double_conv(64, 32)
        self.upconv1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = self.double_conv(32, 16)
        self.final_conv = nn.Conv2d(16, output_channels, 1)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        bottleneck = self.bottleneck(self.pool3(enc3))
        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        output = torch.sigmoid(self.final_conv(dec1))
        return output


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

    def create_datasets(self, train_samples, val_samples, test_samples, batch_size, multi_task=True):
        print("📸 Creating training dataset with sample previews...")
        self.train_loader = DataLoader(
            WolffiaSyntheticDataset(
                train_samples, 
                128, 
                real_backgrounds_dir='images',
                save_previews=True,  # FIXED: Enable sample preview saving
                preview_path='sample_preview'
            ), 
            batch_size=batch_size, 
            shuffle=True
        )
        
        print("📸 Creating validation dataset...")
        self.val_loader = DataLoader(
            WolffiaSyntheticDataset(val_samples, 128, real_backgrounds_dir='images'), 
            batch_size=batch_size
        )
        
        print("📸 Creating test dataset...")
        self.test_loader = DataLoader(
            WolffiaSyntheticDataset(test_samples, 128, real_backgrounds_dir='images'), 
            batch_size=batch_size
        )
        
        print("✅ Datasets created! Training samples will be saved to 'sample_preview/' directory")

    def initialize_model(self, base_filters=32, multi_task=True):
        self.model = WolffiaCNN().to(self.device)

    def train_model(self, epochs=50, learning_rate=0.001):
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        best_loss = float('inf')
        self.history = {'losses': [], 'val_accuracies': [], 'epochs': epochs, 'best_val_loss': float('inf')}

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for images, masks in self.train_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(self.train_loader)
            self.history['losses'].append(avg_loss)
            scheduler.step(avg_loss)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

            # Always save best model only
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch + 1,
                    'loss': best_loss
                }, 'models/wolffia_cnn_best.pth')
                print(f"💾 Best model updated at epoch {epoch+1} with loss {best_loss:.4f}")
                self.history['best_val_loss'] = best_loss

        return self.history

    def evaluate_model(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for images, masks in self.test_loader:
                images, masks = images.to(self.device), masks.to(self.device)
                outputs = self.model(images)
                predictions = (outputs > 0.5).float()
                correct += (predictions == masks).sum().item()
                total += masks.numel()
        accuracy = correct / total
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
    def __init__(self, num_samples=10000, image_size=128, real_backgrounds_dir='images', save_previews=False, preview_path='sample_preview'):
        self.num_samples = num_samples
        self.image_size = image_size
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