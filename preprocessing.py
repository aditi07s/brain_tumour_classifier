"""
Preprocessing module for MRI scans
Includes resizing, normalization, and denoising
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import restoration
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class Preprocessor:
    """Preprocess MRI scans."""
    
    def __init__(self, target_shape: Tuple[int, int, int] = None):
        """
        Initialize preprocessor.
        
        Args:
            target_shape: Target shape for resizing (H, W, D). If None, no resizing.
        """
        self.target_shape = target_shape
    
    def normalize(self, volume: np.ndarray, per_channel: bool = True) -> np.ndarray:
        """
        Normalize volume to [0, 1] range.
        
        Args:
            volume: Input volume (H, W, D) or (H, W, D, C)
            per_channel: Normalize each channel independently
            
        Returns:
            Normalized volume
        """
        volume = volume.copy().astype(np.float32)
        
        if per_channel and volume.ndim == 4:
            # Normalize each modality independently
            for c in range(volume.shape[-1]):
                channel = volume[..., c]
                v_min, v_max = np.percentile(channel[channel > 0], [0.5, 99.5])
                if v_max > v_min:
                    volume[..., c] = np.clip((channel - v_min) / (v_max - v_min), 0, 1)
        else:
            # Global normalization
            v_min, v_max = np.percentile(volume[volume > 0], [0.5, 99.5])
            if v_max > v_min:
                volume = np.clip((volume - v_min) / (v_max - v_min), 0, 1)
        
        logger.debug(f"Normalized volume: min={volume.min():.4f}, max={volume.max():.4f}")
        return volume
    
    def denoise(self, volume: np.ndarray, method: str = 'bilateral') -> np.ndarray:
        """
        Remove noise from volume.
        
        Args:
            volume: Input volume
            method: 'bilateral', 'tv', or 'gaussian'
            
        Returns:
            Denoised volume
        """
        volume = volume.copy().astype(np.float32)
        
        if method == 'bilateral':
            # TV (Total Variation) denoising - good for preserving edges
            volume = restoration.denoise_tv_chambolle(volume, weight=0.1, eps=0.002)
        elif method == 'gaussian':
            # Simple Gaussian filter
            volume = gaussian_filter(volume, sigma=0.8)
        elif method == 'tv':
            volume = restoration.denoise_tv_chambolle(volume, weight=0.1)
        
        logger.debug(f"Denoised volume using {method} method")
        return volume
    
    def resize(self, volume: np.ndarray, seg: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resize volume to target shape using interpolation.
        
        Args:
            volume: Input volume (H, W, D) or (H, W, D, C)
            seg: Optional segmentation volume
            
        Returns:
            Tuple of (resized_volume, resized_seg or None)
        """
        if self.target_shape is None:
            return volume, seg
        
        from scipy.ndimage import zoom
        
        current_shape = volume.shape[:3]
        zoom_factors = np.array(self.target_shape) / np.array(current_shape)
        
        # Resize volume
        if volume.ndim == 4:
            # Handle multi-channel
            resized = np.zeros((self.target_shape + (volume.shape[-1],)), dtype=np.float32)
            for c in range(volume.shape[-1]):
                resized[..., c] = zoom(volume[..., c], zoom_factors, order=1)
        else:
            resized = zoom(volume, zoom_factors, order=1)
        
        # Resize segmentation if provided
        resized_seg = None
        if seg is not None:
            resized_seg = zoom(seg, zoom_factors, order=0).astype(seg.dtype)
        
        logger.debug(f"Resized volume from {current_shape} to {self.target_shape}")
        return resized, resized_seg
    
    def preprocess(self, volume: np.ndarray, seg: np.ndarray = None,
                   normalize: bool = True, denoise: bool = True,
                   resize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete preprocessing pipeline.
        
        Args:
            volume: Input volume
            seg: Optional segmentation
            normalize: Whether to normalize
            denoise: Whether to denoise
            resize: Whether to resize
            
        Returns:
            Tuple of (processed_volume, processed_seg or None)
        """
        if denoise:
            volume = self.denoise(volume, method='tv')
        
        if resize and self.target_shape:
            volume, seg = self.resize(volume, seg)
        
        if normalize:
            volume = self.normalize(volume)
        
        return volume, seg
