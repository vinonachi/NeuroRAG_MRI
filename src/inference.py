import torch

def load_model(model_class, model_path, device="cpu"):
    
    model = model_class().to(device)

    try:
        checkpoint = torch.load(model_path, map_location=device)

        # If saved as state_dict
        if isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)

        # If saved as full model
        else:
            model = checkpoint

        model.eval()

    except Exception as e:
        print("Model loading failed:", e)

    return model
