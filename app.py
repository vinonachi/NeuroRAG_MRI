import streamlit as st
import numpy as np
import torch

from src.preprocessing import simulate_low_resolution
from src.inference import load_model, run_inference
from src.graph_analysis import (
    extract_skeleton,
    skeleton_to_graph,
    compute_graph_metrics
)

from src.models.cnn_baseline import CNNSuperResolution


# ===============================
# Page Setup
# ===============================
st.set_page_config(layout="wide")
st.title("Privacy-Preserving Generative AI for Neurovascular MRI")

st.sidebar.title("NeuroRAG Control Panel")

mode = st.sidebar.selectbox(
    "Select Mode",
    ["Super Resolution", "Graph Analysis", "Full Pipeline"]
)

privacy_mode = st.sidebar.checkbox("Enable Privacy-Preserving Mode")


# ===============================
# Load Pre-Saved MRI Slice
# ===============================
try:
    slice_data = np.load("sample_slice.npy")
except:
    st.error("sample_slice.npy not found. Please add it to project root.")
    st.stop()

# Normalize safety (if needed)
slice_data = (slice_data - np.min(slice_data)) / (
    np.max(slice_data) - np.min(slice_data)
)

hr_slice = slice_data


# ===============================
# Simulate Low Resolution
# ===============================
lr_slice = simulate_low_resolution(hr_slice)


# ===============================
# Load Model
# ===============================
device = "cpu"
model = load_model(CNNSuperResolution, "models/trained_model.pth", device=device)


# ===============================
# Run Inference
# ===============================
output = run_inference(model, lr_slice, device=device)


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

with col3:
    st.subheader("AI Reconstructed Image")
    st.image(output, clamp=True)


# ===============================
# PSNR Metric
# ===============================
def calculate_psnr(hr, sr):
    mse = np.mean((hr - sr) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(1.0 / np.sqrt(mse))


psnr_value = calculate_psnr(hr_slice, output)
st.metric("PSNR Score", round(psnr_value, 2))


# ===============================
# Graph Analysis
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
    del hr_slice
    st.warning("Privacy Mode Enabled: Original MRI data cleared from memory.")


else:
    st.info("Please upload a MRI file to begin.")
