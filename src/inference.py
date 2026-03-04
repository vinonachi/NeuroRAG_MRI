import torch
import numpy as np


def load_model(model_class, model_path, device="cpu"):

    model = model_class().to(device)

    try:
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.eval()
    except:
        print("Model loading failed")

    return model


def run_inference(model, image, device="cpu"):

    # convert numpy -> tensor
    img = torch.tensor(image, dtype=torch.float32)

    # shape correction
    img = img.unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)

    output = output.squeeze().cpu().numpy()

    # normalize output
    output = (output - output.min()) / (output.max() - output.min() + 1e-8)

    return output
