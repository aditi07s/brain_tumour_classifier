"""
Visualization utilities for results
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Visualizer:
    """Visualize MRI scans and predictions."""
    
    # Class color mapping for BraTS
    CLASS_NAMES = {
        0: 'Background',
        1: 'Necrotic',
        2: 'Edema',
        3: 'Enhancing'
    }
    
    CLASS_COLORS = {
        0: [0, 0, 0],        # Black - background
        1: [0, 0, 1],        # Blue - necrotic
        2: [0, 1, 0],        # Green - edema
        3: [1, 0, 0],        # Red - enhancing
    }
    
    def __init__(self, output_dir: str = 'results'):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory to save visualizations
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_slices(self, volume: np.ndarray, modality_idx: int = 0,
                   num_slices: int = 9, title: str = 'MRI Slices') -> plt.Figure:
        """
        Plot multiple slices from a volume.
        
        Args:
            volume: Input volume (D, H, W, C) or (D, H, W)
            modality_idx: Which modality to display (if multi-channel)
            num_slices: Number of slices to display
            title: Figure title
            
        Returns:
            Matplotlib figure
        """
        if volume.ndim == 4:
            volume = volume[..., modality_idx]
        
        d, h, w = volume.shape
        slice_indices = np.linspace(0, d-1, num_slices, dtype=int)
        
        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, slice_idx in enumerate(slice_indices):
            ax = axes[idx]
            ax.imshow(volume[slice_idx, :, :], cmap='gray')
            ax.set_title(f'Slice {slice_idx}')
            ax.axis('off')
        
        fig.suptitle(title)
        return fig
    
    def plot_segmentation_overlay(self, volume: np.ndarray,
                                 segmentation: np.ndarray,
                                 modality_idx: int = 0,
                                 num_slices: int = 9,
                                 title: str = 'Segmentation Overlay',
                                 alpha: float = 0.3) -> plt.Figure:
        """
        Plot segmentation overlaid on MRI.
        
        Args:
            volume: MRI volume (D, H, W, C)
            segmentation: Segmentation (D, H, W)
            modality_idx: Which modality to use
            num_slices: Number of slices
            title: Figure title
            alpha: Overlay transparency
            
        Returns:
            Matplotlib figure
        """
        if volume.ndim == 4:
            mri = volume[..., modality_idx]
        else:
            mri = volume
        
        d, h, w = mri.shape
        slice_indices = np.linspace(0, d-1, num_slices, dtype=int)
        
        fig, axes = plt.subplots(3, 3, figsize=(12, 10))
        axes = axes.flatten()
        
        for idx, slice_idx in enumerate(slice_indices):
            ax = axes[idx]
            
            # Show MRI
            ax.imshow(mri[slice_idx, :, :], cmap='gray')
            
            # Overlay segmentation
            seg_slice = segmentation[slice_idx, :, :]
            if seg_slice.max() > 0:
                # Create colored overlay
                overlay = self._create_colored_overlay(seg_slice)
                ax.imshow(overlay, alpha=alpha)
            
            ax.set_title(f'Slice {slice_idx}')
            ax.axis('off')
        
        fig.suptitle(title)
        return fig
    
    def plot_prediction_comparison(self, volume: np.ndarray,
                                  gt_seg: np.ndarray,
                                  pred_seg: np.ndarray,
                                  modality_idx: int = 0,
                                  num_slices: int = 9) -> plt.Figure:
        """
        Compare ground truth and prediction.
        
        Args:
            volume: MRI volume
            gt_seg: Ground truth segmentation
            pred_seg: Predicted segmentation
            modality_idx: Which modality
            num_slices: Number of slices
            
        Returns:
            Matplotlib figure
        """
        if volume.ndim == 4:
            mri = volume[..., modality_idx]
        else:
            mri = volume
        
        d, h, w = mri.shape
        slice_indices = np.linspace(0, d-1, num_slices, dtype=int)
        
        fig, axes = plt.subplots(num_slices, 3, figsize=(15, 5*num_slices))
        
        for row, slice_idx in enumerate(slice_indices):
            # MRI
            axes[row, 0].imshow(mri[slice_idx, :, :], cmap='gray')
            axes[row, 0].set_title(f'MRI Slice {slice_idx}')
            axes[row, 0].axis('off')
            
            # Ground truth
            gt_overlay = self._create_colored_overlay(gt_seg[slice_idx, :, :])
            axes[row, 1].imshow(mri[slice_idx, :, :], cmap='gray')
            axes[row, 1].imshow(gt_overlay, alpha=0.4)
            axes[row, 1].set_title(f'Ground Truth {slice_idx}')
            axes[row, 1].axis('off')
            
            # Prediction
            pred_overlay = self._create_colored_overlay(pred_seg[slice_idx, :, :])
            axes[row, 2].imshow(mri[slice_idx, :, :], cmap='gray')
            axes[row, 2].imshow(pred_overlay, alpha=0.4)
            axes[row, 2].set_title(f'Prediction {slice_idx}')
            axes[row, 2].axis('off')
        
        fig.suptitle('Segmentation Comparison: Ground Truth vs Prediction')
        return fig
    
    def plot_training_history(self, history: dict, save_path: str = None) -> plt.Figure:
        """
        Plot training history.
        
        Args:
            history: Training history dict with 'loss', 'accuracy', etc.
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Loss
        if 'loss' in history:
            axes[0, 0].plot(history['loss'], label='Train Loss')
            if 'val_loss' in history:
                axes[0, 0].plot(history['val_loss'], label='Val Loss')
            axes[0, 0].set_title('Loss')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Accuracy
        if 'accuracy' in history:
            axes[0, 1].plot(history['accuracy'], label='Train Accuracy')
            if 'val_accuracy' in history:
                axes[0, 1].plot(history['val_accuracy'], label='Val Accuracy')
            axes[0, 1].set_title('Accuracy')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Precision
        if 'precision' in history:
            axes[1, 0].plot(history['precision'], label='Train Precision')
            if 'val_precision' in history:
                axes[1, 0].plot(history['val_precision'], label='Val Precision')
            axes[1, 0].set_title('Precision')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Recall
        if 'recall' in history:
            axes[1, 1].plot(history['recall'], label='Train Recall')
            if 'val_recall' in history:
                axes[1, 1].plot(history['val_recall'], label='Val Recall')
            axes[1, 1].set_title('Recall')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        fig.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved training history plot to {save_path}")
        
        return fig
    
    def _create_colored_overlay(self, segmentation: np.ndarray) -> np.ndarray:
        """
        Create RGB overlay from segmentation labels.
        
        Args:
            segmentation: Segmentation (H, W) with class indices
            
        Returns:
            RGB image (H, W, 3)
        """
        h, w = segmentation.shape
        overlay = np.zeros((h, w, 3), dtype=np.uint8)
        
        for class_id, color in self.CLASS_COLORS.items():
            mask = segmentation == class_id
            overlay[mask] = np.array(color) * 255
        
        return overlay
    
    def save_figure(self, fig: plt.Figure, filename: str):
        """Save figure to disk."""
        path = self.output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved figure to {path}")
        plt.close(fig)
