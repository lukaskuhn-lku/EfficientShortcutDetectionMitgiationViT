import os
import torch
from tqdm import tqdm
import numpy as np
from surgeon_pytorch import Inspect
from safetensors import save_file
import gc


def get_predictions(
    loader,
    loader_name,
    model,
    model_name,
    layer_names,
    W_K,
    conv_out_model,
    device="cpu",
):
    # skip if already exists
    if os.path.exists(
        f"./outputs/{model_name}/{loader_name}_activations.safetensors"
    ) and os.path.exists(f"./outputs/{model_name}/{loader_name}_targets.safetensors"):
        print(f"{loader_name} already exists. Skipping...")
        return (
            f"./outputs/{model_name}/{loader_name}_activations.safetensors",
            f"./outputs/{model_name}/{loader_name}_targets.safetensors",
            f"./outputs/{model_name}/{loader_name}_logits.safetensors",
        )

    # Create an Extract model that only computes the desired intermediate activations

    layers = layer_names

    model_ext = Inspect(model, layer=layers)

    # Get the intermediate activations
    all_activations = {}
    all_targets = []
    all_logits = []
    all_keys = []

    with torch.no_grad():
        for images, targets in tqdm(loader):
            # Original image
            images_orig = images.to(device)
            all_targets.append(targets)

            activations_tta = []

            # Run original image through the model
            logits, activations = model_ext(images_orig)
            out = conv_out_model(images_orig)
            all_logits.append(logits.cpu().numpy())

            ln_1 = out[:, 1:, :]
            k = ln_1 @ W_K.T

            k = k.reshape(len(images), 196, 12, 64)
            k_mean = k.mean(dim=2)

            all_keys.append(k_mean.cpu().numpy())

            activations_tta.append([act.cpu().numpy() for act in activations])

            # Mean over the patch tokens (excluding the CLS token), except for the final layer
            for i, acts in enumerate(zip(*activations_tta)):
                layer_name = layer_names[i]
                if layer_name not in all_activations:
                    all_activations[layer_name] = []

                acts = acts[0]

                if (
                    len(acts[0].shape) > 1
                ):  # Check if the activations have more than 2 dimensions
                    mean_patch_activations = [act.mean(axis=0) for act in acts]
                    all_activations[layer_name].append(
                        np.stack(mean_patch_activations, axis=0)
                    )

            # Clear the GPU memory
            del images_orig, activations_tta
            torch.cuda.empty_cache()

    # Concatenate the results for each layer
    for layer_name in all_activations:
        all_activations[layer_name] = np.concatenate(
            all_activations[layer_name], axis=0
        )

    # Save all_activations as safetensor file
    all_activations = {
        layer_name: torch.tensor(activations).contiguous()
        for layer_name, activations in all_activations.items()
    }
    targets = {"targets": torch.cat(all_targets, dim=0)}
    all_logits = {"logits": torch.tensor(np.concatenate(all_logits, axis=0))}
    all_keys = {"keys": torch.tensor(np.concatenate(all_keys, axis=0))}

    os.makedirs(f"outputs/{model_name}", exist_ok=True)
    save_file(
        all_activations, f"./outputs/{model_name}/{loader_name}_activations.safetensors"
    )
    save_file(targets, f"./outputs/{model_name}/{loader_name}_targets.safetensors")
    save_file(all_logits, f"./outputs/{model_name}/{loader_name}_logits.safetensors")
    save_file(all_keys, f"./outputs/{model_name}/{loader_name}_keys.safetensors")

    del all_activations, targets
    torch.cuda.empty_cache()
    gc.collect()

    return (
        f"./outputs/{model_name}/{loader_name}_activations.safetensors",
        f"./outputs/{model_name}/{loader_name}_targets.safetensors",
        f"./outputs/{model_name}/{loader_name}_logits.safetensors",
    )
