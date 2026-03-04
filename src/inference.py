import torch
import numpy as np


def load_model(model_class, model_path, device="cpu"):

    model = model_class().to(device)

    try:
        checkpoint = torch.load(model_path, map_location=device)

        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)

        model.eval()

    except:
        print("Model loading failed")

    return model


def run_inference(model, image, device="cpu"):

    img = torch.tensor(image).unsqueeze(0).unsqueeze(0).float().to(device)

    with torch.no_grad():
        output = model(img)

    output = output.squeeze().cpu().numpy()

    return output
