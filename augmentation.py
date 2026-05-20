"""
3D Data Augmentation
Rotation, flipping, zooming for training data
"""

import numpy as np
from scipy.ndimage import rotate, zoom
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class Augmentor3D:
    """Apply 3D augmentations to MRI patches."""
    
    def __init__(self, rotation_angles: list = None, flip_axes: list = None, 
                 zoom_range: Tuple[float, float] = None):
        """
        Initialize augmentor.
        
        Args:
            rotation_angles: List of rotation angles to sample from
            flip_axes: List of axes to flip along
            zoom_range: Tuple of (min_zoom, max_zoom)
        """
        self.rotation_angles = rotation_angles or [10, 20, 30]
        self.flip_axes = flip_axes or [0, 1, 2]
        self.zoom_range = zoom_range or (0.8, 1.2)
    
    def rotate_3d(self, patch: np.ndarray, max_angle: float = 15) -> np.ndarray:
        """
        Randomly rotate patch in 3D space.
        
        Args:
            patch: Input patch (D, H, W) or (D, H, W, C)
            max_angle: Maximum rotation angle in degrees
            
        Returns:
            Rotated patch
        """
        angle = np.random.uniform(-max_angle, max_angle)
        axis = np.random.choice([0, 1, 2])
        order = 1 if patch.ndim == 4 else 1  # Linear interpolation
        
        if patch.ndim == 4:
            # Rotate each channel
            rotated = np.zeros_like(patch)
            for c in range(patch.shape[-1]):
                rotated[..., c] = rotate(patch[..., c], angle, axes=(axis, (axis + 1) % 3),
                                        order=order, reshape=False)
        else:
            rotated = rotate(patch, angle, axes=(axis, (axis + 1) % 3),
                           order=order, reshape=False)
        
        return rotated
    
    def flip_3d(self, patch: np.ndarray) -> np.ndarray:
        """
        Randomly flip patch along one axis.
        
        Args:
            patch: Input patch
            
        Returns:
            Flipped patch
        """
        axis = np.random.choice(self.flip_axes)
        return np.flip(patch, axis=axis).copy()
    
    def zoom_3d(self, patch: np.ndarray) -> np.ndarray:
        """
        Randomly zoom patch.
        
        Args:
            patch: Input patch
            
        Returns:
            Zoomed patch
        """
        z = np.random.uniform(self.zoom_range[0], self.zoom_range[1])
        order = 1 if patch.ndim == 4 else 1
        
        if patch.ndim == 4:
            # Zoom each channel
            zoomed = np.zeros_like(patch)
            for c in range(patch.shape[-1]):
                zoomed[..., c] = zoom(patch[..., c], z, order=order)
        else:
            zoomed = zoom(patch, z, order=order)
        
        # Pad or crop to maintain shape
        original_shape = patch.shape
        zoomed_shape = zoomed.shape
        
        if zoom > 1.0:  # Cropped - center crop
            slices = tuple(slice(int((s - o) / 2), int((s - o) / 2) + o)
                          for s, o in zip(zoomed_shape, original_shape))
            zoomed = zoomed[slices]
        elif zoom < 1.0:  # Padded - center pad
            padding = tuple((int((o - s) / 2), int((o - s) / 2))
                           for s, o in zip(zoomed_shape, original_shape))
            zoomed = np.pad(zoomed, padding, mode='constant', constant_values=0)
        
        return zoomed
    
    def augment(self, patch: np.ndarray, seg_patch: np.ndarray = None,
                p_rotation: float = 0.5, p_flip: float = 0.5,
                p_zoom: float = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random augmentations to patch and corresponding segmentation.
        
        Args:
            patch: Input patch
            seg_patch: Optional segmentation patch
            p_rotation: Probability of rotation
            p_flip: Probability of flip
            p_zoom: Probability of zoom
            
        Returns:
            Tuple of (augmented_patch, augmented_seg or None)
        """
        augmented_patch = patch.copy()
        augmented_seg = seg_patch.copy() if seg_patch is not None else None
        
        if np.random.rand() < p_rotation:
            augmented_patch = self.rotate_3d(augmented_patch)
            if augmented_seg is not None:
                augmented_seg = self.rotate_3d(augmented_seg.astype(np.float32))
        
        if np.random.rand() < p_flip:
            augmented_patch = self.flip_3d(augmented_patch)
            if augmented_seg is not None:
                augmented_seg = self.flip_3d(augmented_seg.astype(np.float32))
        
        if np.random.rand() < p_zoom:
            augmented_patch = self.zoom_3d(augmented_patch)
            if augmented_seg is not None:
                augmented_seg = self.zoom_3d(augmented_seg.astype(np.float32))
        
        if augmented_seg is not None:
            augmented_seg = augmented_seg.astype(seg_patch.dtype)
        
        return augmented_patch, augmented_seg
