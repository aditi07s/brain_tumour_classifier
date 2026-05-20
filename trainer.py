"""
Training utilities for 3D U-Net model
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, List

logger = logging.getLogger(__name__)

class Trainer:
    """Handle model training with checkpointing and early stopping."""
    
    def __init__(self, model: keras.Model, checkpoint_dir: str = 'models/checkpoints'):
        """
        Initialize trainer.
        
        Args:
            model: Keras model to train
            checkpoint_dir: Directory for saving checkpoints
        """
        self.model = model
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = None
        self.best_loss = float('inf')
    
    def get_callbacks(self, monitor: str = 'val_loss',
                     patience: int = 10) -> List[keras.callbacks.Callback]:
        """
        Get training callbacks.
        
        Args:
            monitor: Metric to monitor
            patience: Patience for early stopping
            
        Returns:
            List of callbacks
        """
        callbacks = [
            keras.callbacks.ModelCheckpoint(
                str(self.checkpoint_dir / 'best_model.h5'),
                monitor=monitor,
                save_best_only=True,
                verbose=1
            ),
            keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor=monitor,
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            keras.callbacks.TensorBoard(
                log_dir=str(self.checkpoint_dir / 'logs'),
                histogram_freq=1,
                profile_batch=0
            )
        ]
        return callbacks
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray,
             x_val: np.ndarray, y_val: np.ndarray,
             epochs: int = 100, batch_size: int = 8,
             validation_split: float = None) -> keras.callbacks.History:
        """
        Train the model.
        
        Args:
            x_train: Training input patches
            y_train: Training target segmentations
            x_val: Validation input patches
            y_val: Validation target segmentations
            epochs: Number of epochs
            batch_size: Batch size
            validation_split: Alternative to x_val/y_val
            
        Returns:
            Training history
        """
        logger.info(f"Starting training for {epochs} epochs with batch size {batch_size}")
        logger.info(f"Training data shape: {x_train.shape}")
        logger.info(f"Validation data shape: {x_val.shape if x_val is not None else 'None'}")
        
        callbacks = self.get_callbacks()
        
        # Ensure shapes are compatible
        x_train = x_train.astype(np.float32)
        y_train = y_train.astype(np.float32)
        
        if x_val is not None:
            x_val = x_val.astype(np.float32)
            y_val = y_val.astype(np.float32)
            validation_data = (x_val, y_val)
        else:
            validation_data = validation_split
        
        self.history = self.model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info(f"Training completed. Final loss: {self.history.history['loss'][-1]:.4f}")
        return self.history
    
    def save_model(self, path: str):
        """Save model to disk."""
        self.model.save(path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model from disk."""
        self.model = keras.models.load_model(path)
        logger.info(f"Model loaded from {path}")

def create_data_generator(batch_size: int = 8, augment: bool = True):
    """
    Create a data generator for training.
    
    Args:
        batch_size: Batch size
        augment: Whether to apply augmentation
        
    Returns:
        ImageDataGenerator
    """
    if augment:
        gen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='nearest'
        )
    else:
        gen = keras.preprocessing.image.ImageDataGenerator()
    
    return gen

class DataGenerator(keras.utils.Sequence):
    """Custom data generator for 3D patches."""
    
    def __init__(self, patches: List[np.ndarray], 
                segmentations: List[np.ndarray],
                batch_size: int = 8, shuffle: bool = True,
                augment: bool = False):
        """
        Initialize generator.
        
        Args:
            patches: List of patches
            segmentations: List of segmentations
            batch_size: Batch size
            shuffle: Whether to shuffle
            augment: Whether to augment
        """
        self.patches = patches
        self.segmentations = segmentations
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augment = augment
        self.indices = np.arange(len(patches))
        
        if shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.patches) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        
        batch_patches = np.array([self.patches[i] for i in batch_indices])
        batch_segs = np.array([self.segmentations[i] for i in batch_indices])
        
        return batch_patches.astype(np.float32), batch_segs.astype(np.float32)
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
