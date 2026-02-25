import torch
import numpy as np


def load_model(model_class, model_path, device="cpu"):
    model = model_class().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def run_inference(model, image, device="cpu"):
    input_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)

    output = output.squeeze().cpu().numpy()
    return output
