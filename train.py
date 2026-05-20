"""
Main training script
"""

import argparse
import numpy as np
from pathlib import Path
import logging

from src.config import Config
from src.data_loader import BraTSDataLoader
from src.preprocessing import Preprocessor
from src.patch_extractor import PatchExtractor
from src.augmentation import Augmentor3D
from src.model import create_model
from src.trainer import Trainer
from src.evaluator import Evaluator
from src.visualizer import Visualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(config: Config, num_samples: int = None):
    """Load and preprocess BraTS data."""
    logger.info("Loading BraTS dataset...")
    
    loader = BraTSDataLoader(config.get('data.raw_dir'))
    
    if num_samples is None:
        num_samples = loader.get_num_scans()
    
    preprocessor = Preprocessor(target_shape=None)  # No resizing initially
    all_patches = []
    all_segs = []
    
    for idx in range(min(num_samples, loader.get_num_scans())):
        logger.info(f"Processing scan {idx+1}/{min(num_samples, loader.get_num_scans())}")
        
        volume, seg = loader.load_scan(idx)
        logger.info(f"  Raw shape: {volume.shape}")
        
        # Preprocess
        volume, seg = preprocessor.preprocess(volume, seg, 
                                             normalize=True, 
                                             denoise=True,
                                             resize=False)
        
        # Extract patches
        patch_size = config.get('data.patch_size', 64)
        extractor = PatchExtractor(patch_size=patch_size, 
                                  stride=config.get('data.stride', 32))
        
        patches, segs = extractor.extract_all_patches(volume, seg, include_empty=False)
        
        logger.info(f"  Extracted {len(patches)} patches")
        
        all_patches.extend(patches)
        all_segs.extend(segs)
    
    return np.array(all_patches), np.array(all_segs)

def one_hot_encode(seg: np.ndarray, num_classes: int = 4) -> np.ndarray:
    """Convert class indices to one-hot encoding."""
    seg_one_hot = np.zeros((seg.shape[0], seg.shape[1], seg.shape[2], num_classes))
    for c in range(num_classes):
        seg_one_hot[..., c] = (seg == c).astype(int)
    return seg_one_hot

def train(args):
    """Train the model."""
    config = Config(args.config) if args.config else Config()
    
    logger.info("=" * 60)
    logger.info("Brain Tumor Classifier - Training")
    logger.info("=" * 60)
    
    # Load data
    patches, segs = load_and_preprocess_data(config, num_samples=args.num_samples)
    
    # Convert to one-hot
    num_classes = config.get('model.output_classes', 4)
    segs_one_hot = one_hot_encode(segs, num_classes)
    
    # Split data
    test_split = config.get('training.test_split', 0.15)
    val_split = config.get('training.validation_split', 0.15)
    
    n_total = len(patches)
    n_test = int(n_total * test_split)
    n_val = int((n_total - n_test) * val_split)
    n_train = n_total - n_test - n_val
    
    indices = np.random.permutation(n_total)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train+n_val]
    test_idx = indices[n_train+n_val:]
    
    x_train, y_train = patches[train_idx], segs_one_hot[train_idx]
    x_val, y_val = patches[val_idx], segs_one_hot[val_idx]
    x_test, y_test = patches[test_idx], segs_one_hot[test_idx]
    
    logger.info(f"\nData split:")
    logger.info(f"  Training: {len(x_train)}")
    logger.info(f"  Validation: {len(x_val)}")
    logger.info(f"  Testing: {len(x_test)}")
    
    # Build model
    logger.info("\nBuilding model...")
    patch_size = config.get('data.patch_size', 64)
    input_shape = (patch_size, patch_size, patch_size, config.get('model.input_channels', 4))
    
    model = create_model(input_shape=input_shape,
                        num_classes=num_classes,
                        depth=config.get('model.depth', 4),
                        filters_base=config.get('model.filters_base', 32),
                        learning_rate=config.get('training.learning_rate', 1e-3))
    
    model.summary()
    
    # Train
    logger.info("\nStarting training...")
    trainer = Trainer(model, checkpoint_dir=config.get('paths.models_dir'))
    
    history = trainer.train(
        x_train, y_train,
        x_val, y_val,
        epochs=config.get('training.epochs', 100),
        batch_size=config.get('training.batch_size', 8)
    )
    
    # Save model
    model_path = Path(config.get('paths.models_dir')) / 'final_model.h5'
    trainer.save_model(str(model_path))
    
    # Evaluate
    logger.info("\nEvaluating model...")
    evaluator = Evaluator(num_classes=num_classes)
    
    # Predictions on test set
    y_pred = model.predict(x_test, verbose=1)
    
    metrics = evaluator.compute_metrics(y_test, y_pred)
    per_class_metrics = evaluator.compute_per_class_metrics(y_test, y_pred)
    
    logger.info("\nOverall Metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    logger.info("\nPer-class Metrics:")
    for class_id, class_metrics in per_class_metrics.items():
        logger.info(f"  {class_id}:")
        for metric_name, value in class_metrics.items():
            logger.info(f"    {metric_name}: {value:.4f}")
    
    # Visualize
    logger.info("\nVisualizing results...")
    visualizer = Visualizer(output_dir=config.get('paths.results_dir'))
    
    # Plot training history
    hist_dict = {
        'loss': history.history.get('loss', []),
        'val_loss': history.history.get('val_loss', []),
        'accuracy': history.history.get('accuracy', []),
        'val_accuracy': history.history.get('val_accuracy', []),
    }
    
    fig = visualizer.plot_training_history(hist_dict, 
                                          save_path=Path(config.get('paths.results_dir')) / 'training_history.png')
    
    logger.info("\n" + "=" * 60)
    logger.info("Training completed!")
    logger.info("=" * 60)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train brain tumor classifier')
    parser.add_argument('--config', type=str, default=None, help='Config file path')
    parser.add_argument('--num-samples', type=int, default=None, help='Number of samples to use')
    
    args = parser.parse_args()
    train(args)
