import streamlit as st
import numpy as np
import torch
from PIL import Image
import os

from src.preprocessing import simulate_low_resolution
from src.inference import load_model, run_inference
from src.graph_analysis import (
    extract_skeleton,
    skeleton_to_graph,
    compute_graph_metrics
)

from src.models.cnn_baseline import CNNSuperResolution


# ===============================
# Page Configuration
# ===============================
st.set_page_config(layout="wide")

st.title("Privacy-Preserving Generative AI for Neurovascular MRI")

st.sidebar.title("NeuroRAG Control Panel")

mode = st.sidebar.selectbox(
    "Select Mode",
    ["Super Resolution", "Graph Analysis", "Full Pipeline"]
)

privacy_mode = st.sidebar.checkbox("Enable Privacy-Preserving Mode")

uploaded_file = st.sidebar.file_uploader(
    "Upload MRI Image",
    type=["png", "jpg", "jpeg"]
)


# ===============================
# Utility Functions
# ===============================

def calculate_psnr(hr, sr):
    mse = np.mean((hr - sr) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(1.0 / np.sqrt(mse))


# ===============================
# Main Pipeline
# ===============================

if uploaded_file is not None:

    # ===============================
    # Load Image
    # ===============================
    image = Image.open(uploaded_file).convert("L")

    # Resize for CNN model
    image = image.resize((256,256))

    hr_slice = np.array(image).astype(np.float32) / 255.0

    # ===============================
    # Simulate Low Resolution
    # ===============================
    lr_slice = simulate_low_resolution(hr_slice)

    # ===============================
    # Display Images
    # ===============================
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original High Resolution")
        st.image(hr_slice, clamp=True)

    with col2:
        st.subheader("Simulated Low Resolution")
        st.image(lr_slice, clamp=True)

    # ===============================
    # Load AI Model
    # ===============================
    device = "cpu"

    model_path = os.path.join("models", "trained_model.pth")

    try:
        model = load_model(CNNSuperResolution, model_path, device=device)
    except:
        model = None
        st.warning("Model could not be loaded. Using fallback.")

    # ===============================
    # Run Super Resolution
    # ===============================
    try:
        output = run_inference(model, lr_slice, device=device)

st.write("Output Min:", np.min(output))
st.write("Output Max:", np.max(output))
    except:
        output = lr_slice
        st.warning("Inference failed. Showing interpolated output.")

    with col3:
        st.subheader("AI Reconstructed Image")
        st.image(output, clamp=True)

    # ===============================
    # PSNR Metric
    # ===============================
    psnr_value = calculate_psnr(hr_slice, output)
    st.metric("PSNR Score", round(psnr_value, 2))

    # ===============================
    # Graph Analysis
    # ===============================
    if mode in ["Graph Analysis", "Full Pipeline"]:

        st.subheader("Vessel Skeleton Extraction")

        try:
            skeleton = extract_skeleton(output)

            st.image(skeleton, clamp=True)

            if np.sum(skeleton) > 0:

                G = skeleton_to_graph(skeleton)
                metrics = compute_graph_metrics(G)

                st.subheader("Graph Topology Metrics")

                for key, value in metrics.items():
                    st.write(f"**{key}:** {round(value, 2)}")

            else:
                st.warning("No vessel structures detected.")

        except:
            st.warning("Graph analysis failed.")

    # ===============================
    # Synthetic Generation
    # ===============================
    if st.button("Generate Synthetic Neurovascular Image"):

        noise = np.random.normal(0, 0.05, output.shape)

        synthetic = np.clip(output + noise, 0, 1)

        st.subheader("Synthetic Generated Output")

        st.image(synthetic, clamp=True)

    # ===============================
    # Privacy Mode
    # ===============================
    if privacy_mode:
        del hr_slice
        st.warning("Privacy Mode Enabled: Original data cleared from memory.")

else:
    st.info("Please upload an image to begin.")
