"""
Evaluation script for testing model performance
"""

import argparse
import numpy as np
from pathlib import Path
import logging
import tensorflow as tf

from src.data_loader import BraTSDataLoader
from src.preprocessing import Preprocessor
from src.patch_extractor import PatchExtractor
from src.predictor import Predictor
from src.evaluator import Evaluator, ConfusionMatrixCalculator
from src.visualizer import Visualizer
from src.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(model_path: str, data_dir: str = None, num_samples: int = 5):
    """Evaluate model on test set."""
    config = Config()
    
    logger.info("=" * 60)
    logger.info("Brain Tumor Classifier - Evaluation")
    logger.info("=" * 60)
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    # Load data
    if data_dir is None:
        data_dir = config.get('data.raw_dir')
    
    logger.info(f"Loading data from {data_dir}")
    loader = BraTSDataLoader(data_dir)
    
    # Process samples
    preprocessor = Preprocessor()
    patch_size = config.get('data.patch_size', 64)
    extractor = PatchExtractor(patch_size=patch_size, 
                              stride=config.get('data.stride', 32))
    predictor = Predictor(model)
    evaluator = Evaluator(num_classes=config.get('model.output_classes', 4))
    visualizer = Visualizer(output_dir=config.get('paths.results_dir'))
    
    all_y_true = []
    all_y_pred = []
    
    for idx in range(min(num_samples, loader.get_num_scans())):
        logger.info(f"\nProcessing sample {idx+1}/{min(num_samples, loader.get_num_scans())}")
        
        # Load volume
        volume, seg = loader.load_scan(idx)
        
        # Preprocess
        volume, seg = preprocessor.preprocess(volume, seg,
                                             normalize=True,
                                             denoise=True,
                                             resize=False)
        
        # Extract patches
        patches, segs = extractor.extract_all_patches(volume, seg, include_empty=False)
        patches = np.array(patches).astype(np.float32)
        segs = np.array(segs).astype(np.int32)
        
        # Predict
        logger.info(f"Predicting on {len(patches)} patches...")
        predictions = predictor.predict_patches(patches, batch_size=4)
        
        # Convert to one-hot
        segs_one_hot = np.zeros((segs.shape[0], segs.shape[1], segs.shape[2], 
                                segs.shape[3], config.get('model.output_classes', 4)))
        for c in range(config.get('model.output_classes', 4)):
            segs_one_hot[..., c] = (segs == c).astype(int)
        
        all_y_true.append(segs_one_hot)
        all_y_pred.append(predictions)
        
        # Visualize sample prediction
        if idx == 0:
            center_idx = len(patches) // 2
            pred = predictions[center_idx]
            seg = segs[center_idx]
            
            # Plot comparison
            fig = visualizer.plot_segmentation_overlay(
                patches[center_idx],
                seg,
                modality_idx=0,
                num_slices=1,
                title=f'Prediction vs Ground Truth - Scan {idx}'
            )
            visualizer.save_figure(fig, f'sample_prediction_{idx}.png')
    
    # Aggregate metrics
    y_true = np.concatenate(all_y_true, axis=0)
    y_pred = np.concatenate(all_y_pred, axis=0)
    
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Metrics")
    logger.info("=" * 60)
    
    # Overall metrics
    metrics = evaluator.compute_metrics(y_true, y_pred)
    logger.info("\nOverall Metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # Per-class metrics
    per_class_metrics = evaluator.compute_per_class_metrics(y_true, y_pred)
    logger.info("\nPer-class Metrics:")
    for class_name, class_metrics in per_class_metrics.items():
        logger.info(f"  {class_name}:")
        for metric_name, value in class_metrics.items():
            logger.info(f"    {metric_name}: {value:.4f}")
    
    # Dice scores
    dice_scores = evaluator.compute_dice_score(y_true, y_pred)
    logger.info("\nDice Scores:")
    for class_name, dice in dice_scores.items():
        logger.info(f"  {class_name}: {dice:.4f}")
    
    # IoU scores
    iou_scores = evaluator.compute_iou(y_true, y_pred)
    logger.info("\nIoU Scores:")
    for class_name, iou in iou_scores.items():
        logger.info(f"  {class_name}: {iou:.4f}")
    
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation completed!")
    logger.info("=" * 60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate brain tumor classifier')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--data', type=str, default=None, help='Path to data directory')
    parser.add_argument('--num-samples', type=int, default=5, help='Number of samples')
    
    args = parser.parse_args()
    evaluate_model(args.model, args.data, args.num_samples)
