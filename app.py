import streamlit as st
import numpy as np
import torch
import matplotlib.pyplot as plt
import nibabel as nib
from skimage.transform import resize

from src.models.cnn_baseline import CNNSuperResolution

st.title("Privacy-Preserving Neurovascular MRI Reconstruction")

st.sidebar.header("Upload MRI File")

uploaded_file = st.sidebar.file_uploader("Upload .nii or .nii.gz", type=["nii", "gz"])

def normalize(volume):
    return (volume - np.min(volume)) / (np.max(volume) - np.min(volume))

def simulate_low_resolution(img, scale=2):
    h, w = img.shape
    low_res = resize(img, (h//scale, w//scale))
    low_res = resize(low_res, (h, w))
    return low_res

if uploaded_file is not None:

    img = nib.load(uploaded_file)
    volume = img.get_fdata()
    volume = normalize(volume)

    slice_index = volume.shape[2] // 2
    hr_slice = volume[:, :, slice_index]
    lr_slice = simulate_low_resolution(hr_slice)

    st.subheader("Original High Resolution")
    st.image(hr_slice, clamp=True)

    st.subheader("Simulated Low Resolution")
    st.image(lr_slice, clamp=True)

    # Load trained model
    model = CNNSuperResolution()
    model.load_state_dict(torch.load("models/trained_model.pth", map_location="cpu"))
    model.eval()

    input_tensor = torch.tensor(lr_slice, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor).squeeze().numpy()

    st.subheader("AI Reconstructed High Resolution")
    st.image(output, clamp=True)

    st.success("Reconstruction Completed Successfully")
