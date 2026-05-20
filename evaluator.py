"""
Model Evaluation and Metrics
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import logging

logger = logging.getLogger(__name__)

class Evaluator:
    """Evaluate model performance."""
    
    def __init__(self, num_classes: int = 4):
        """
        Initialize evaluator.
        
        Args:
            num_classes: Number of output classes
        """
        self.num_classes = num_classes
        self.metrics = {}
    
    def compute_metrics(self, y_true: np.ndarray, y_pred: np.ndarray,
                       average: str = 'weighted') -> dict:
        """
        Compute classification metrics.
        
        Args:
            y_true: Ground truth labels (N, H, W, D, C) or flattened
            y_pred: Predicted labels (N, H, W, D, C) or flattened
            average: Averaging method ('weighted', 'macro', 'micro')
            
        Returns:
            Dictionary of metrics
        """
        # Flatten predictions and ground truth
        y_true_flat = y_true.argmax(axis=-1).flatten() if y_true.ndim > 1 else y_true.flatten()
        y_pred_flat = y_pred.argmax(axis=-1).flatten() if y_pred.ndim > 1 else y_pred.flatten()
        
        metrics = {
            'accuracy': accuracy_score(y_true_flat, y_pred_flat),
            'precision': precision_score(y_true_flat, y_pred_flat, 
                                        average=average, zero_division=0),
            'recall': recall_score(y_true_flat, y_pred_flat, 
                                  average=average, zero_division=0),
            'f1': f1_score(y_true_flat, y_pred_flat, 
                          average=average, zero_division=0),
        }
        
        self.metrics.update(metrics)
        return metrics
    
    def compute_per_class_metrics(self, y_true: np.ndarray, 
                                 y_pred: np.ndarray) -> dict:
        """
        Compute per-class metrics.
        
        Args:
            y_true: Ground truth (N, H, W, D, C)
            y_pred: Predictions (N, H, W, D, C)
            
        Returns:
            Dictionary of per-class metrics
        """
        # Flatten
        y_true_flat = y_true.argmax(axis=-1).flatten()
        y_pred_flat = y_pred.argmax(axis=-1).flatten()
        
        metrics = {}
        for class_id in range(self.num_classes):
            y_true_binary = (y_true_flat == class_id).astype(int)
            y_pred_binary = (y_pred_flat == class_id).astype(int)
            
            metrics[f'class_{class_id}'] = {
                'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
                'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
                'f1': f1_score(y_true_binary, y_pred_binary, zero_division=0),
            }
        
        return metrics
    
    def compute_dice_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Compute Dice coefficient (useful for segmentation).
        
        Args:
            y_true: Ground truth (N, H, W, D, C)
            y_pred: Predictions (N, H, W, D, C)
            
        Returns:
            Dictionary of Dice scores per class
        """
        y_true_labels = y_true.argmax(axis=-1)
        y_pred_labels = y_pred.argmax(axis=-1)
        
        dice_scores = {}
        for class_id in range(self.num_classes):
            intersection = np.sum((y_true_labels == class_id) & (y_pred_labels == class_id))
            union = np.sum(y_true_labels == class_id) + np.sum(y_pred_labels == class_id)
            
            if union == 0:
                dice = 1.0 if intersection == 0 else 0.0
            else:
                dice = 2.0 * intersection / union
            
            dice_scores[f'class_{class_id}'] = dice
        
        dice_scores['mean'] = np.mean(list(dice_scores.values()))
        return dice_scores
    
    def compute_iou(self, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Compute Intersection over Union (IoU).
        
        Args:
            y_true: Ground truth
            y_pred: Predictions
            
        Returns:
            Dictionary of IoU scores per class
        """
        y_true_labels = y_true.argmax(axis=-1)
        y_pred_labels = y_pred.argmax(axis=-1)
        
        iou_scores = {}
        for class_id in range(self.num_classes):
            intersection = np.sum((y_true_labels == class_id) & (y_pred_labels == class_id))
            union = np.sum((y_true_labels == class_id) | (y_pred_labels == class_id))
            
            if union == 0:
                iou = 1.0 if intersection == 0 else 0.0
            else:
                iou = intersection / union
            
            iou_scores[f'class_{class_id}'] = iou
        
        iou_scores['mean'] = np.mean(list(iou_scores.values()))
        return iou_scores
    
    def print_summary(self):
        """Print evaluation summary."""
        logger.info("=" * 50)
        logger.info("Evaluation Summary")
        logger.info("=" * 50)
        for key, value in self.metrics.items():
            if isinstance(value, dict):
                logger.info(f"\n{key}:")
                for k, v in value.items():
                    logger.info(f"  {k}: {v:.4f}")
            else:
                logger.info(f"{key}: {value:.4f}")
        logger.info("=" * 50)

class ConfusionMatrixCalculator:
    """Calculate confusion matrix for segmentation."""
    
    @staticmethod
    def compute(y_true: np.ndarray, y_pred: np.ndarray,
               num_classes: int) -> np.ndarray:
        """
        Compute confusion matrix.
        
        Args:
            y_true: Ground truth (one-hot or class indices)
            y_pred: Predictions (one-hot or class indices)
            num_classes: Number of classes
            
        Returns:
            Confusion matrix (num_classes x num_classes)
        """
        # Convert to class indices if needed
        if y_true.ndim > 1 and y_true.shape[-1] > 1:
            y_true = y_true.argmax(axis=-1)
        if y_pred.ndim > 1 and y_pred.shape[-1] > 1:
            y_pred = y_pred.argmax(axis=-1)
        
        # Flatten
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # Compute confusion matrix
        cm = np.zeros((num_classes, num_classes), dtype=np.int32)
        for i in range(len(y_true_flat)):
            cm[y_true_flat[i], y_pred_flat[i]] += 1
        
        return cm
