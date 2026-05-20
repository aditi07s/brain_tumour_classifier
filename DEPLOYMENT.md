# Deployment Guide

## 🌐 Streamlit Cloud Deployment

### Files
- **`requirements.txt`** - Cloud-compatible (uses standard TensorFlow for Linux)
- **`requirements-dev.txt`** - Local development (Mac M2 with GPU acceleration)

### Deploy Steps

1. **Update your GitHub repo:**
   ```bash
   git add requirements.txt
   git commit -m "Update requirements for Streamlit Cloud"
   git push origin main
   ```

2. **Go to Streamlit Cloud:**
   https://share.streamlit.io/

3. **Click "New app":**
   - Repository: Your GitHub repo
   - Branch: `main`
   - Main file path: `app/streamlit_app.py`

4. **Wait for deployment** (2-3 minutes)

### If Still Getting Errors

The app needs a trained model. Add this to `app/streamlit_app.py`:

```python
import streamlit as st

@st.cache_resource
def get_sample_model():
    """If no model exists, show message."""
    model_dir = Path(__file__).parent.parent / 'models'
    model_files = list(model_dir.glob('*.h5'))
    
    if not model_files:
        st.error("No trained model found. Please train locally first or upload to cloud storage.")
        st.stop()
    
    return model_files[0]
```

### ✅ Local Development (Mac M2)

Use the dev requirements:
```bash
pip install -r requirements-dev.txt
python train.py --num-samples 5
streamlit run app/streamlit_app.py
```

### ☁️ Cloud Deployment (Linux/Streamlit Cloud)

Uses standard `requirements.txt`:
- Standard `tensorflow` (CPU - works on Streamlit Cloud)
- No `tensorflow-macos` or `tensorflow-metal`
- All other packages compatible

### 📝 Note

The cloud version runs on CPU (no GPU), so:
- Inference is slower (but acceptable for demo)
- Use smaller models or pre-trained weights
- Consider uploading pre-trained model to cloud storage

For production with GPU, use:
- AWS EC2 with GPU
- Google Cloud Run
- Azure Container Instances
- Custom server with Mac M2
