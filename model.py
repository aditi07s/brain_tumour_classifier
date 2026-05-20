"""
3D U-Net Model Architecture
Patch-based 3D CNN for tumor segmentation
"""

import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models

logger = logging.getLogger(__name__)

class UNet3D:
    """3D U-Net architecture for medical image segmentation."""
    
    def __init__(self, input_shape: tuple = (64, 64, 64, 4),
                 num_classes: int = 4, depth: int = 4,
                 filters_base: int = 32):
        """
        Initialize 3D U-Net.
        
        Args:
            input_shape: Input shape (D, H, W, C)
            num_classes: Number of output classes
            depth: Number of encoding/decoding levels
            filters_base: Base number of filters
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.depth = depth
        self.filters_base = filters_base
        self.model = None
    
    def conv_block(self, x, filters, kernel_size=3, activation='relu'):
        """
        Double convolution block.
        
        Args:
            x: Input tensor
            filters: Number of filters
            kernel_size: Kernel size
            activation: Activation function
            
        Returns:
            Output tensor
        """
        x = layers.Conv3D(filters, kernel_size, padding='same',
                         activation=activation, kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv3D(filters, kernel_size, padding='same',
                         activation=activation, kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        return x
    
    def build(self):
        """Build the U-Net model."""
        inputs = keras.Input(shape=self.input_shape)
        
        # Encoder
        encoder_blocks = []
        x = inputs
        filters = self.filters_base
        
        for level in range(self.depth):
            logger.debug(f"Encoder level {level}: filters={filters}")
            
            # Convolution block
            x = self.conv_block(x, filters)
            encoder_blocks.append(x)
            
            # Pooling
            if level < self.depth - 1:
                x = layers.MaxPooling3D((2, 2, 2))(x)
            
            filters *= 2
        
        # Decoder
        for level in range(self.depth - 2, -1, -1):
            filters //= 2
            logger.debug(f"Decoder level {level}: filters={filters}")
            
            # Upsampling
            x = layers.Conv3DTranspose(filters, (2, 2, 2), strides=(2, 2, 2),
                                      padding='same')(x)
            
            # Skip connection
            skip = encoder_blocks[level]
            x = layers.Concatenate()([x, skip])
            
            # Convolution block
            x = self.conv_block(x, filters)
        
        # Output layer
        outputs = layers.Conv3D(self.num_classes, 1, padding='same',
                              activation='softmax')(x)
        
        self.model = models.Model(inputs, outputs)
        logger.info(f"Built U-Net3D with {len(self.model.layers)} layers")
        return self.model
    
    def compile(self, learning_rate: float = 1e-3):
        """
        Compile the model.
        
        Args:
            learning_rate: Learning rate for optimizer
        """
        if self.model is None:
            self.build()
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 
                    keras.metrics.Precision(),
                    keras.metrics.Recall()]
        )
        
        logger.info(f"Compiled model with learning rate={learning_rate}")
    
    def get_model(self):
        """Get the compiled model."""
        if self.model is None:
            self.build()
            self.compile()
        return self.model
    
    def summary(self):
        """Print model summary."""
        if self.model is None:
            self.build()
        self.model.summary()

def create_model(input_shape=(64, 64, 64, 4), num_classes=4,
                depth=4, filters_base=32, learning_rate=1e-3):
    """
    Convenience function to create and compile U-Net3D.
    
    Args:
        input_shape: Input shape
        num_classes: Number of classes
        depth: Network depth
        filters_base: Base filters
        learning_rate: Learning rate
        
    Returns:
        Compiled Keras model
    """
    unet = UNet3D(input_shape, num_classes, depth, filters_base)
    unet.build()
    unet.compile(learning_rate)
    return unet.get_model()
