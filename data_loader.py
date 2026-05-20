"""
Data Loader for BraTS 2024 Dataset
Handles loading .nii/.nii.gz files and converting to numpy arrays
"""

import nibabel as nib
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BraTSDataLoader:
    """Load and manage BraTS 2024 MRI scans."""
    
    MODALITIES = {
        't1': '_t1.nii.gz',
        't1ce': '_t1ce.nii.gz',
        't2': '_t2.nii.gz',
        'flair': '_flair.nii.gz',
    }
    
    SEG_SUFFIX = '_seg.nii.gz'
    
    def __init__(self, data_dir: str):
        """
        Initialize data loader.
        
        Args:
            data_dir: Path to BraTS dataset directory
        """
        self.data_dir = Path(data_dir)
        if not self.data_dir.exists():
            raise ValueError(f"Data directory does not exist: {data_dir}")
        
        self.scans = []
        self._discover_scans()
    
    def _discover_scans(self):
        """Discover all scan folders in the dataset."""
        for folder in sorted(self.data_dir.glob('BraTS*_*_*')):
            if folder.is_dir():
                self.scans.append(folder)
        
        logger.info(f"Discovered {len(self.scans)} scans")
    
    def load_scan(self, scan_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a single scan with all modalities stacked.
        
        Args:
            scan_id: Index of the scan to load
            
        Returns:
            Tuple of (volume, segmentation)
            - volume: shape (H, W, D, 4) - 4 modalities
            - segmentation: shape (H, W, D) - tumor labels
        """
        scan_dir = self.scans[scan_id]
        logger.info(f"Loading scan: {scan_dir.name}")
        
        # Load all modalities
        modalities_data = []
        for mod_name, mod_suffix in self.MODALITIES.items():
            mod_file = scan_dir / f"{scan_dir.name}{mod_suffix}"
            if not mod_file.exists():
                raise FileNotFoundError(f"Modality file not found: {mod_file}")
            
            img = nib.load(mod_file)
            data = img.get_fdata().astype(np.float32)
            modalities_data.append(data)
            logger.debug(f"  Loaded {mod_name}: {data.shape}")
        
        # Stack modalities: (H, W, D) x 4 -> (H, W, D, 4)
        volume = np.stack(modalities_data, axis=-1)
        
        # Load segmentation
        seg_file = scan_dir / f"{scan_dir.name}{self.SEG_SUFFIX}"
        if not seg_file.exists():
            raise FileNotFoundError(f"Segmentation file not found: {seg_file}")
        
        seg_img = nib.load(seg_file)
        segmentation = seg_img.get_fdata().astype(np.int32)
        
        logger.info(f"Loaded volume shape: {volume.shape}, segmentation shape: {segmentation.shape}")
        
        return volume, segmentation
    
    def load_by_name(self, scan_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load a scan by its folder name."""
        scan_dir = self.data_dir / scan_name
        if not scan_dir.exists():
            raise ValueError(f"Scan folder not found: {scan_name}")
        
        idx = self.scans.index(scan_dir)
        return self.load_scan(idx)
    
    def get_scan_names(self) -> List[str]:
        """Get all scan folder names."""
        return [scan.name for scan in self.scans]
    
    def get_num_scans(self) -> int:
        """Get total number of scans."""
        return len(self.scans)
    
    def __len__(self):
        return len(self.scans)
    
    def __getitem__(self, idx):
        return self.load_scan(idx)
