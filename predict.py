"""
Prediction script for generating predictions on new scans
"""

import argparse
import numpy as np
from pathlib import Path
import logging
import tensorflow as tf
import nibabel as nib

from src.data_loader import BraTSDataLoader
from src.preprocessing import Preprocessor
from src.predictor import Predictor
from src.visualizer import Visualizer
from src.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def predict_on_scan(model_path: str, scan_path: str, output_dir: str = 'results'):
    """Predict tumor segmentation on a single scan."""
    config = Config()
    
    logger.info("=" * 60)
    logger.info("Brain Tumor Classifier - Prediction")
    logger.info("=" * 60)
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Load scan
    logger.info(f"Loading scan from {scan_path}")
    scan_path = Path(scan_path)
    
    if scan_path.is_file() and scan_path.suffix in ['.nii', '.gz']:
        # Single file
        img = nib.load(scan_path)
        volume = img.get_fdata().astype(np.float32)
        
        # Add modality dimension if needed
        if volume.ndim == 3:
            volume = np.stack([volume] * 4, axis=-1)
        elif volume.ndim == 4 and volume.shape[-1] == 1:
            volume = np.concatenate([volume] * 4, axis=-1)
    elif scan_path.is_dir():
        # BraTS format folder
        loader = BraTSDataLoader(scan_path.parent)
        scan_name = scan_path.name
        volume, _ = loader.load_by_name(scan_name)
    else:
        raise ValueError(f"Invalid path: {scan_path}")
    
    logger.info(f"Loaded volume shape: {volume.shape}")
    
    # Preprocess
    logger.info("Preprocessing...")
    preprocessor = Preprocessor()
    volume, _ = preprocessor.preprocess(volume, seg=None,
                                       normalize=True,
                                       denoise=True,
                                       resize=False)
    
    # Predict
    logger.info("Running inference...")
    predictor = Predictor(model)
    
    patch_size = config.get('data.patch_size', 64)
    stride = config.get('data.stride', 32)
    
    prediction = predictor.predict_volume(volume, patch_size=patch_size,
                                        stride=stride, use_average=True)
    
    # Get segmentation
    segmentation = predictor.get_segmentation(prediction)
    confidence = predictor.get_confidence(prediction)
    
    logger.info(f"Prediction shape: {prediction.shape}")
    logger.info(f"Segmentation shape: {segmentation.shape}")
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save segmentation as NIfTI
    seg_nii = nib.Nifti1Image(segmentation, np.eye(4))
    seg_path = output_dir / 'segmentation.nii.gz'
    nib.save(seg_nii, seg_path)
    logger.info(f"Saved segmentation to {seg_path}")
    
    # Save confidence
    conf_nii = nib.Nifti1Image(confidence, np.eye(4))
    conf_path = output_dir / 'confidence.nii.gz'
    nib.save(conf_nii, conf_path)
    logger.info(f"Saved confidence to {conf_path}")
    
    # Visualize
    logger.info("Generating visualizations...")
    visualizer = Visualizer(output_dir=str(output_dir))
    
    # Plot slices
    fig = visualizer.plot_slices(volume, modality_idx=0, num_slices=9,
                                title='Input MRI')
    visualizer.save_figure(fig, 'input_slices.png')
    
    # Plot segmentation overlay
    fig = visualizer.plot_segmentation_overlay(volume, segmentation,
                                              modality_idx=0, num_slices=9,
                                              title='Tumor Segmentation',
                                              alpha=0.5)
    visualizer.save_figure(fig, 'segmentation_overlay.png')
    
    # Print statistics
    logger.info("\n" + "=" * 60)
    logger.info("Segmentation Statistics")
    logger.info("=" * 60)
    
    unique_classes, counts = np.unique(segmentation, return_counts=True)
    total_voxels = segmentation.size
    
    class_names = {
        0: 'Background',
        1: 'Necrotic',
        2: 'Edema',
        3: 'Enhancing'
    }
    
    for class_id, count in zip(unique_classes, counts):
        percentage = (count / total_voxels) * 100
        class_name = class_names.get(class_id, f'Class {class_id}')
        logger.info(f"{class_name:12} ({class_id}): {percentage:6.2f}% ({count:,} voxels)")
    
    logger.info(f"\nMean Confidence: {confidence.mean():.4f}")
    logger.info(f"Confidence Range: [{confidence.min():.4f}, {confidence.max():.4f}]")
    
    logger.info("\n" + "=" * 60)
    logger.info("Prediction completed!")
    logger.info("=" * 60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict tumor segmentation')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--input', type=str, required=True, help='Path to input scan')
    parser.add_argument('--output', type=str, default='results', help='Output directory')
    
    args = parser.parse_args()
    predict_on_scan(args.model, args.input, args.output)
