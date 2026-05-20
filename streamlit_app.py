"""
Streamlit web interface for tumor classification
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import BraTSDataLoader
from src.preprocessing import Preprocessor
from src.predictor import Predictor
from src.visualizer import Visualizer
from src.config import Config

# Page configuration
st.set_page_config(
    page_title="Brain Tumor Classifier",
    page_icon="🧠",
    layout="wide"
)

# Title and description
st.title("🧠 Brain Tumor Classifier")
st.markdown("""
    This application uses a 3D U-Net deep learning model trained on the BraTS 2024 dataset
    to classify and segment brain tumors from MRI scans.
    
    **Features:**
    - Upload MRI scans (.nii, .nii.gz files)
    - Automatic preprocessing and normalization
    - Tumor segmentation using 3D deep learning
    - Interactive visualization of results
    """)

# Sidebar configuration
st.sidebar.title("Configuration")
config = Config()

# Model selection
model_dir = Path(config.get('paths.models_dir', 'models'))
model_files = list(model_dir.glob('*.h5'))

if model_files:
    selected_model = st.sidebar.selectbox(
        "Select Model",
        model_files,
        format_func=lambda x: x.name
    )
else:
    st.error("No trained models found. Please train a model first using train.py")
    st.stop()

# Load model
@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(str(model_path))

model = load_model(selected_model)
st.sidebar.success(f"Model loaded: {selected_model.name}")

# Processing parameters
st.sidebar.markdown("### Processing Parameters")
patch_size = st.sidebar.slider("Patch Size", 32, 128, 64, step=32)
stride = st.sidebar.slider("Stride", 16, 64, 32, step=16)
normalize = st.sidebar.checkbox("Normalize Input", value=True)
denoise = st.sidebar.checkbox("Denoise Input", value=True)

# Main interface
tab1, tab2, tab3 = st.tabs(["Upload & Predict", "Sample Dataset", "About"])

with tab1:
    st.header("Upload MRI Scan")
    
    uploaded_files = st.file_uploader(
        "Select MRI files (.nii or .nii.gz)",
        type=['nii', 'gz'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.info(f"Uploaded {len(uploaded_files)} file(s)")
        
        if st.button("Process & Predict", key="predict_btn"):
            try:
                # Create temporary directory for uploads
                import tempfile
                import nibabel as nib
                
                with tempfile.TemporaryDirectory() as tmpdir:
                    # Save uploaded files
                    for uploaded_file in uploaded_files:
                        with open(f"{tmpdir}/{uploaded_file.name}", "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    
                    st.write("Processing...")
                    
                    # Load and preprocess
                    preprocessor = Preprocessor()
                    
                    # Load first file as example
                    nib_img = nib.load(f"{tmpdir}/{uploaded_files[0].name}")
                    volume = nib_img.get_fdata().astype(np.float32)
                    
                    # Preprocess
                    if denoise:
                        volume = preprocessor.denoise(volume)
                    
                    if normalize:
                        volume = preprocessor.normalize(volume)
                    
                    # Add channel dimension if needed
                    if volume.ndim == 3:
                        volume = np.stack([volume] * 4, axis=-1)
                    
                    st.success("Preprocessing completed!")
                    
                    # Predict
                    with st.spinner("Running prediction..."):
                        predictor = Predictor(model)
                        prediction = predictor.predict_volume(
                            volume,
                            patch_size=patch_size,
                            stride=stride
                        )
                    
                    st.success("Prediction completed!")
                    
                    # Get segmentation
                    segmentation = predictor.get_segmentation(prediction)
                    confidence = predictor.get_confidence(prediction)
                    
                    # Visualization
                    st.subheader("Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Input MRI (slice 32)**")
                        if volume.ndim == 4:
                            st.image(volume[32, :, :, 0], use_column_width=True, 
                                   caption="T1 Modality", clamp=True)
                        else:
                            st.image(volume[32, :, :], use_column_width=True, clamp=True)
                    
                    with col2:
                        st.write("**Predicted Segmentation (slice 32)**")
                        seg_slice = segmentation[32, :, :]
                        st.image(seg_slice, use_column_width=True, 
                               caption="Tumor Segments", cmap='viridis')
                    
                    # Statistics
                    st.subheader("Statistics")
                    
                    unique_classes, counts = np.unique(segmentation, return_counts=True)
                    
                    class_names = {
                        0: "Background",
                        1: "Necrotic",
                        2: "Edema",
                        3: "Enhancing"
                    }
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    for class_id, count in zip(unique_classes, counts):
                        percentage = (count / segmentation.size) * 100
                        class_name = class_names.get(class_id, f"Class {class_id}")
                        
                        if class_id == 0:
                            col = col1
                        elif class_id == 1:
                            col = col2
                        elif class_id == 2:
                            col = col3
                        else:
                            col = col4
                        
                        with col:
                            st.metric(class_name, f"{percentage:.2f}%")
                    
                    # Mean confidence
                    st.metric("Mean Confidence", f"{confidence.mean():.4f}")
                    
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")

with tab2:
    st.header("Sample from BraTS Dataset")
    
    st.info("Load and view sample scans from the BraTS 2024 dataset")
    
    data_dir = config.get('data.raw_dir', 'data/raw')
    
    if Path(data_dir).exists():
        try:
            loader = BraTSDataLoader(data_dir)
            scan_names = loader.get_scan_names()
            
            if scan_names:
                selected_scan = st.selectbox("Select a scan:", scan_names)
                
                if st.button("Load Sample", key="load_sample"):
                    try:
                        idx = scan_names.index(selected_scan)
                        volume, seg = loader.load_scan(idx)
                        
                        # Display
                        col1, col2, col3, col4 = st.columns(4)
                        
                        slice_idx = volume.shape[0] // 2
                        
                        with col1:
                            st.image(volume[slice_idx, :, :, 0], caption="T1")
                        with col2:
                            st.image(volume[slice_idx, :, :, 1], caption="T1c")
                        with col3:
                            st.image(volume[slice_idx, :, :, 2], caption="T2")
                        with col4:
                            st.image(volume[slice_idx, :, :, 3], caption="FLAIR")
                        
                        st.image(seg[slice_idx, :, :], caption="Segmentation", 
                               use_column_width=True, clamp=True)
                        
                    except Exception as e:
                        st.error(f"Error loading sample: {str(e)}")
            else:
                st.warning("No scans found in dataset directory")
        except Exception as e:
            st.warning(f"Could not load dataset: {str(e)}")
    else:
        st.warning(f"Dataset directory not found: {data_dir}")

with tab3:
    st.header("About This Application")
    
    st.markdown("""
    ### Overview
    This brain tumor classifier uses a 3D U-Net convolutional neural network to segment
    brain tumors in MRI scans from the BraTS (Brain Tumor Segmentation) dataset.
    
    ### Model Architecture
    - **Type**: 3D U-Net
    - **Input**: 64×64×64 or 96×96×96 patches
    - **Modalities**: T1, T1c (T1 contrast-enhanced), T2, FLAIR
    - **Output Classes**: 4 (Background, Necrotic, Edema, Enhancing)
    
    ### BraTS Dataset
    The model is trained on the BraTS 2024 dataset which contains multi-modal MRI scans
    with expert-annotated tumor segmentations.
    
    ### Classes
    1. **Background (0)**: Non-tumor tissue
    2. **Necrotic (1)**: Non-enhancing tumor core
    3. **Edema (2)**: Tumor-associated edema
    4. **Enhancing (3)**: Contrast-enhancing tumor core
    
    ### Processing Pipeline
    1. **Data Loading**: Read .nii/.nii.gz files
    2. **Preprocessing**: Normalize, denoise, and prepare patches
    3. **Segmentation**: 3D U-Net inference on overlapping patches
    4. **Postprocessing**: Reconstruct full volume and visualize results
    
    ### Technologies
    - TensorFlow/Keras
    - Python 3.8+
    - Streamlit
    """)
