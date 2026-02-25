import streamlit as st
import numpy as np
import torch
import nibabel as nib
import matplotlib.pyplot as plt

from src.data_loader import load_mri, normalize, get_middle_slice
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
    "Upload MRI (.nii / .nii.gz)",
    type=["nii", "gz"]
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

    # Load MRI
    volume = load_mri(uploaded_file)
    volume = normalize(volume)
    hr_slice = get_middle_slice(volume)

    # Simulate low resolution
    lr_slice = simulate_low_resolution(hr_slice)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Original High Resolution")
        st.image(hr_slice, clamp=True)

    with col2:
        st.subheader("Simulated Low Resolution")
        st.image(lr_slice, clamp=True)

    # Load model
    device = "cpu"
    model = load_model(CNNSuperResolution, "models/trained_model.pth", device=device)

    # Run inference
    output = run_inference(model, lr_slice, device=device)

    with col3:
        st.subheader("AI Reconstructed Image")
        st.image(output, clamp=True)

    # ===============================
    # PSNR Metric
    # ===============================
    psnr_value = calculate_psnr(hr_slice, output)
    st.metric("PSNR Score", round(psnr_value, 2))


    # ===============================
    # Graph Analysis Mode
    # ===============================
    if mode in ["Graph Analysis", "Full Pipeline"]:

        st.subheader("Vessel Skeleton Extraction")

        skeleton = extract_skeleton(output)
        st.image(skeleton, clamp=True)

        G = skeleton_to_graph(skeleton)
        metrics = compute_graph_metrics(G)

        st.subheader("Graph Topology Metrics")

        for key, value in metrics.items():
            st.write(f"**{key}:** {round(value, 2)}")


    # ===============================
    # Synthetic Generation
    # ===============================
    if st.button("Generate Synthetic Neurovascular Image"):
        noise = np.random.normal(0, 0.05, output.shape)
        synthetic = output + noise
        synthetic = np.clip(synthetic, 0, 1)

        st.subheader("Synthetic Generated Output")
        st.image(synthetic, clamp=True)


    # ===============================
    # Privacy Mode
    # ===============================
    if privacy_mode:
        del volume
        del hr_slice
        st.warning("Privacy Mode Enabled: Original MRI data cleared from memory.")


else:
    st.info("Please upload a MRI file to begin.")
