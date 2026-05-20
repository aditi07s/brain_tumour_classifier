# Brain Tumor Classifier - BraTS 2024

A comprehensive deep learning project for brain tumor classification and segmentation using 3D CNN architecture on the BraTS 2024 dataset.

## Overview

This project implements a patch-based 3D U-Net model to segment brain tumors from multi-modal MRI scans (T1, T1c, T2, FLAIR). The system processes 64×64×64 or 96×96×96 volumetric patches using a deep convolutional neural network optimized for Mac M2 hardware.

## Workflow

```
Medical Dataset Collection
        ↓
Load MRI/CT Scans (.nii/.nii.gz)
        ↓
Preprocessing
(Resize + Normalize + Remove Noise)
        ↓
Convert Scans into 3D Volume
        ↓
Data Augmentation
(Rotate + Flip + Zoom)
        ↓
Split Dataset
(Train / Validation / Test)
        ↓
Build 3D U-Net Model
        ↓
Train Model on Mac M2
(TensorFlow Metal GPU support)
        ↓
Model Learning
(Adjust weights using loss function)
        ↓
Evaluate Performance
(Accuracy, Precision, Recall, Dice, IoU)
        ↓
Generate Prediction
        ↓
Visualize Results
(Heatmap / Detection Area)
        ↓
Deploy Streamlit Interface
        ↓
Final Output
```

## Project Structure

```
tumour/
├── data/
│   ├── raw/                        # BraTS 2024 raw downloads
│   ├── processed/                  # Preprocessed volumes
│   └── splits/                     # Train/val/test splits
├── models/
│   ├── checkpoints/                # Training checkpoints
│   ├── best_model.h5              # Best model during training
│   └── final_model.h5             # Final trained model
├── results/                        # Visualizations and metrics
├── notebooks/                      # Jupyter notebooks
├── src/                            # Main source code
│   ├── config.py                   # Configuration management
│   ├── data_loader.py              # Load .nii/.nii.gz files
│   ├── preprocessing.py            # Normalize, denoise, resize
│   ├── augmentation.py             # 3D augmentation (rotate, flip, zoom)
│   ├── patch_extractor.py          # Extract 3D patches
│   ├── model.py                    # 3D U-Net architecture
│   ├── trainer.py                  # Training loop and callbacks
│   ├── evaluator.py                # Metrics (accuracy, precision, recall, Dice, IoU)
│   ├── predictor.py                # Inference and predictions
│   └── visualizer.py               # Visualization (heatmaps, slices)
├── app/
│   └── streamlit_app.py            # Web UI for predictions
├── train.py                        # Training entry point
├── evaluate.py                     # Evaluation entry point
├── predict.py                      # Prediction entry point
├── requirements.txt                # Python dependencies
├── config.yaml                     # Training configuration
└── README.md                       # This file
```

## Installation

### 1. Clone the Repository
```bash
cd /Users/aditi/Documents/tumour
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### Note on Mac M2
TensorFlow Metal enables GPU acceleration on Apple Silicon:
```bash
# Already included in requirements.txt
pip install tensorflow-macos>=2.13.0
pip install tensorflow-metal>=1.0.0
```

## Configuration

Edit `config.yaml` to customize:
- **Data paths**: raw_dir, processed_dir
- **Patch size**: 64×64×64 or 96×96×96
- **Training**: batch_size, epochs, learning_rate
- **Augmentation**: rotation, flip, zoom ranges
- **Model**: depth, filters_base, architecture

Default configuration is hardcoded in `src/config.py`.

## Usage

### 1. Prepare BraTS Dataset

Download BraTS 2024 dataset and extract to `data/raw/`:
```
data/raw/
├── BraTS2024_GLI_00XXX_000/
│   ├── BraTS2024_GLI_00XXX_000_t1.nii.gz
│   ├── BraTS2024_GLI_00XXX_000_t1ce.nii.gz
│   ├── BraTS2024_GLI_00XXX_000_t2.nii.gz
│   ├── BraTS2024_GLI_00XXX_000_flair.nii.gz
│   └── BraTS2024_GLI_00XXX_000_seg.nii.gz
├── BraTS2024_GLI_00XXX_001/
└── ...
```

### 2. Train Model

```bash
python train.py --num-samples 10
```

Options:
- `--config`: Path to config.yaml
- `--num-samples`: Number of scans to use (default: all)

### 3. Evaluate Model

```bash
python evaluate.py --model models/final_model.h5
```

### 4. Make Predictions

```bash
python predict.py --model models/final_model.h5 --input data/raw/sample_scan/
```

### 5. Deploy Web Interface

```bash
streamlit run app/streamlit_app.py
```

Then open http://localhost:8501 in your browser.

## Model Architecture

### 3D U-Net

```
Input (64×64×64×4)
    ↓
Encoder:
  Level 0: Conv3D(32) + MaxPool → Conv3D(64)
  Level 1: Conv3D(64) + MaxPool → Conv3D(128)
  Level 2: Conv3D(128) + MaxPool → Conv3D(256)
  Level 3: Conv3D(256) [bottleneck]
    ↓
Decoder:
  Level 2: UpConv + Concatenate + Conv3D(256)
  Level 1: UpConv + Concatenate + Conv3D(128)
  Level 0: UpConv + Concatenate + Conv3D(64)
    ↓
Output Conv3D(4, softmax) → (64×64×64×4)
```

**Features:**
- Skip connections from encoder to decoder
- Batch normalization after each convolution
- He initialization for weights
- Softmax activation for multi-class segmentation

## Training Strategy

1. **Data Loading**: Load 4 modalities (T1, T1c, T2, FLAIR)
2. **Preprocessing**: Normalize per-modality, denoise with TV filter
3. **Patching**: Extract overlapping 64×64×64 patches with stride=32
4. **Augmentation**: Random rotation, flip, zoom on-the-fly
5. **Training**: Adam optimizer with learning rate scheduling
6. **Monitoring**: Checkpointing best model, early stopping
7. **Evaluation**: Metrics on held-out test set

## Performance Metrics

The model is evaluated on:
- **Pixel-level**: Accuracy, Precision, Recall, F1-score
- **Segmentation**: Dice coefficient, IoU (Jaccard Index)
- **Per-class**: Separate metrics for each tumor class

## Visualization

### During Training
- Loss curves (train/val)
- Accuracy curves
- Precision and recall evolution

### During Prediction
- MRI slices with segmentation overlay
- Confidence heatmaps
- Class distribution
- Comparison with ground truth (if available)

## Output Classes

| ID | Name | Color | Description |
|---|---|---|---|
| 0 | Background | Black | Non-tumor tissue |
| 1 | Necrotic | Blue | Non-enhancing tumor core |
| 2 | Edema | Green | Tumor-associated edema |
| 3 | Enhancing | Red | Contrast-enhancing tumor |

## GPU Acceleration

### Mac M2 with TensorFlow Metal
```python
# Automatic with tensorflow-metal installed
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))
```

### Monitor GPU Usage
```bash
# In another terminal
while true; do powermetrics -s cpu_power -n 1; done
```

## Memory Considerations

- **Input patch**: 64×64×64×4 = ~1MB (float32)
- **Batch size 8**: ~8-16MB input + ~16-32MB output
- **Model weights**: ~3-5MB
- **Total GPU memory**: ~100-200MB (M2 has 8GB shared)

## Troubleshooting

### Out of Memory (OOM)
- Reduce batch_size in config.yaml
- Reduce patch_size (smaller patches)
- Use stride=patch_size (non-overlapping patches)

### Slow Training
- Enable TensorFlow Metal (check installation)
- Reduce number of training samples
- Use smaller model (reduce depth or filters_base)

### Data Loading Issues
- Verify .nii.gz files are valid (nibabel can open them)
- Check folder structure matches BraTS format
- Enable debug logging in data_loader.py

## References

- **BraTS Dataset**: https://www.med.upenn.edu/cbica/brats2024/
- **U-Net**: Ronneberger et al., "U-Net: Convolutional Networks for Biomedical Image Segmentation" (2015)
- **3D U-Net**: Çiçek et al., "3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation" (2016)
- **TensorFlow**: https://www.tensorflow.org/
- **Streamlit**: https://streamlit.io/

## Future Enhancements

- [ ] Multi-model ensemble
- [ ] Post-processing refinement (CRF, morphological operations)
- [ ] Uncertainty estimation
- [ ] Real-time prediction with streaming
- [ ] Multi-GPU/TPU training
- [ ] Export to ONNX/TorchScript
- [ ] Medical imaging DICOM support
- [ ] REST API deployment

## License

This project is provided as-is for educational and research purposes.

## Contact

For questions or issues, please refer to the documentation or create an issue in the repository.
