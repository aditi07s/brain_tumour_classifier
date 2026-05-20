"""
Patch Extractor for 3D volumes
Extract 64x64x64 or 96x96x96 patches for training
"""

import numpy as np
from typing import Tuple, List, Generator
import logging

logger = logging.getLogger(__name__)

class PatchExtractor:
    """Extract 3D patches from volumes."""
    
    def __init__(self, patch_size: int = 64, stride: int = 32):
        """
        Initialize patch extractor.
        
        Args:
            patch_size: Size of patches (patch_size x patch_size x patch_size)
            stride: Stride for sliding window
        """
        self.patch_size = patch_size
        self.stride = stride
    
    def extract_patches(self, volume: np.ndarray, seg: np.ndarray = None,
                       include_empty: bool = False) -> Generator:
        """
        Extract patches from a volume using sliding window.
        
        Args:
            volume: Input volume (H, W, D, C)
            seg: Optional segmentation (H, W, D)
            include_empty: Include patches with no foreground
            
        Yields:
            Tuple of (patch, seg_patch) or just patch if seg is None
        """
        h, w, d = volume.shape[:3]
        
        patch_count = 0
        for i in range(0, h - self.patch_size + 1, self.stride):
            for j in range(0, w - self.patch_size + 1, self.stride):
                for k in range(0, d - self.patch_size + 1, self.stride):
                    
                    # Extract patch
                    patch = volume[i:i+self.patch_size, j:j+self.patch_size, 
                                  k:k+self.patch_size]
                    
                    # Check if patch is valid (has foreground)
                    if not include_empty:
                        if patch.ndim == 4:
                            # Use first modality to check foreground
                            foreground = patch[..., 0] > 0
                        else:
                            foreground = patch > 0
                        
                        if np.sum(foreground) < (self.patch_size ** 3) * 0.1:
                            continue  # Skip mostly empty patches
                    
                    seg_patch = None
                    if seg is not None:
                        seg_patch = seg[i:i+self.patch_size, j:j+self.patch_size,
                                       k:k+self.patch_size]
                        yield patch, seg_patch
                    else:
                        yield patch
                    
                    patch_count += 1
        
        logger.info(f"Extracted {patch_count} patches from volume")
    
    def extract_all_patches(self, volume: np.ndarray, seg: np.ndarray = None,
                           include_empty: bool = False) -> Tuple[List, List]:
        """
        Extract all patches at once.
        
        Args:
            volume: Input volume
            seg: Optional segmentation
            include_empty: Include empty patches
            
        Returns:
            Lists of (patches, seg_patches)
        """
        patches = []
        seg_patches = []
        
        for result in self.extract_patches(volume, seg, include_empty):
            if seg is not None:
                patch, seg_patch = result
                patches.append(patch)
                seg_patches.append(seg_patch)
            else:
                patches.append(result)
        
        if seg is not None:
            return patches, seg_patches
        else:
            return patches, None
    
    def extract_center_patch(self, volume: np.ndarray, 
                            seg: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract center patch from volume.
        
        Args:
            volume: Input volume
            seg: Optional segmentation
            
        Returns:
            Tuple of (patch, seg_patch or None)
        """
        h, w, d = volume.shape[:3]
        start_h = (h - self.patch_size) // 2
        start_w = (w - self.patch_size) // 2
        start_d = (d - self.patch_size) // 2
        
        patch = volume[start_h:start_h+self.patch_size,
                      start_w:start_w+self.patch_size,
                      start_d:start_d+self.patch_size]
        
        seg_patch = None
        if seg is not None:
            seg_patch = seg[start_h:start_h+self.patch_size,
                           start_w:start_w+self.patch_size,
                           start_d:start_d+self.patch_size]
        
        return patch, seg_patch
    
    def reconstruct_volume(self, patches: List[np.ndarray],
                          original_shape: Tuple[int, int, int],
                          use_average: bool = True) -> np.ndarray:
        """
        Reconstruct volume from overlapping patches.
        
        Args:
            patches: List of patches
            original_shape: Original volume shape
            use_average: Average overlapping regions or take last
            
        Returns:
            Reconstructed volume
        """
        h, w, d = original_shape
        reconstructed = np.zeros(original_shape)
        counts = np.zeros(original_shape)
        
        patch_idx = 0
        for i in range(0, h - self.patch_size + 1, self.stride):
            for j in range(0, w - self.patch_size + 1, self.stride):
                for k in range(0, d - self.patch_size + 1, self.stride):
                    
                    if patch_idx >= len(patches):
                        break
                    
                    patch = patches[patch_idx]
                    reconstructed[i:i+self.patch_size, j:j+self.patch_size,
                                 k:k+self.patch_size] += patch
                    counts[i:i+self.patch_size, j:j+self.patch_size,
                          k:k+self.patch_size] += 1
                    
                    patch_idx += 1
        
        if use_average:
            reconstructed[counts > 0] /= counts[counts > 0]
        
        return reconstructed
