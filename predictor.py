"""
Prediction module for generating predictions on new data
"""

import numpy as np
import tensorflow as tf
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

class Predictor:
    """Generate predictions using trained model."""
    
    def __init__(self, model: tf.keras.Model):
        """
        Initialize predictor.
        
        Args:
            model: Trained Keras model
        """
        self.model = model
    
    def predict_patch(self, patch: np.ndarray) -> np.ndarray:
        """
        Predict on a single patch.
        
        Args:
            patch: Input patch (D, H, W, C)
            
        Returns:
            Prediction (D, H, W, num_classes)
        """
        # Add batch dimension
        patch_batch = np.expand_dims(patch, axis=0).astype(np.float32)
        prediction = self.model.predict(patch_batch, verbose=0)
        return prediction[0]
    
    def predict_patches(self, patches: np.ndarray, batch_size: int = 4) -> np.ndarray:
        """
        Predict on multiple patches.
        
        Args:
            patches: Input patches (N, D, H, W, C)
            batch_size: Batch size for prediction
            
        Returns:
            Predictions (N, D, H, W, num_classes)
        """
        patches = patches.astype(np.float32)
        predictions = self.model.predict(patches, batch_size=batch_size, verbose=1)
        return predictions
    
    def predict_volume(self, volume: np.ndarray, patch_size: int = 64,
                      stride: int = 32, use_average: bool = True) -> np.ndarray:
        """
        Predict on full volume using sliding window.
        
        Args:
            volume: Input volume (D, H, W, C)
            patch_size: Size of patches
            stride: Stride for sliding window
            use_average: Average overlapping regions
            
        Returns:
            Full volume prediction (D, H, W, num_classes)
        """
        d, h, w, c = volume.shape
        num_classes = self.model.output_shape[-1]
        
        # Initialize output
        predictions = np.zeros((d, h, w, num_classes), dtype=np.float32)
        counts = np.zeros((d, h, w, 1), dtype=np.float32)
        
        # Sliding window prediction
        patch_count = 0
        for i in range(0, d - patch_size + 1, stride):
            for j in range(0, h - patch_size + 1, stride):
                for k in range(0, w - patch_size + 1, stride):
                    
                    # Extract patch
                    patch = volume[i:i+patch_size, j:j+patch_size, k:k+patch_size, :]
                    
                    # Predict
                    patch_pred = self.predict_patch(patch)
                    
                    # Add to output
                    predictions[i:i+patch_size, j:j+patch_size, k:k+patch_size, :] += patch_pred
                    counts[i:i+patch_size, j:j+patch_size, k:k+patch_size, :] += 1
                    
                    patch_count += 1
        
        # Average if requested
        if use_average:
            mask = counts > 0
            predictions[mask.squeeze()] /= counts[mask]
        
        logger.info(f"Predicted {patch_count} patches for volume")
        return predictions
    
    def get_segmentation(self, prediction: np.ndarray) -> np.ndarray:
        """
        Get hard segmentation from soft prediction.
        
        Args:
            prediction: Soft prediction (D, H, W, num_classes)
            
        Returns:
            Hard segmentation (D, H, W)
        """
        return prediction.argmax(axis=-1).astype(np.int32)
    
    def get_confidence(self, prediction: np.ndarray) -> np.ndarray:
        """
        Get confidence map from prediction.
        
        Args:
            prediction: Soft prediction (D, H, W, num_classes)
            
        Returns:
            Confidence map (D, H, W) - max probability at each voxel
        """
        return prediction.max(axis=-1)

class EnsemblePredictor:
    """Ensemble predictions from multiple models."""
    
    def __init__(self, models: list):
        """
        Initialize ensemble.
        
        Args:
            models: List of trained models
        """
        self.models = models
        self.predictors = [Predictor(model) for model in models]
    
    def predict_patch(self, patch: np.ndarray, method: str = 'average') -> np.ndarray:
        """
        Predict on patch using ensemble.
        
        Args:
            patch: Input patch
            method: 'average', 'max', or 'voting'
            
        Returns:
            Ensemble prediction
        """
        predictions = np.array([p.predict_patch(patch) for p in self.predictors])
        
        if method == 'average':
            return predictions.mean(axis=0)
        elif method == 'max':
            return predictions.max(axis=0)
        elif method == 'voting':
            segmentations = predictions.argmax(axis=-1)
            # Simple majority voting
            result = np.zeros_like(segmentations[0])
            for idx in np.ndindex(result.shape):
                result[idx] = np.bincount(segmentations[:, idx]).argmax()
            return result
        else:
            raise ValueError(f"Unknown method: {method}")
