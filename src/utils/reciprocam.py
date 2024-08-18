import os
import torch
import numpy as np
from tqdm import tqdm
from safetensors import save_file
from torch.nn import functional as F
from surgeon_pytorch import Inspect


# Function to generate Gaussian mask
def generate_gaussian_mask(size, sigma=1.0):
    """Generate a 2D Gaussian mask."""
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    d = np.sqrt(x * x + y * y)
    g = np.exp(-(d**2 / (2.0 * sigma**2)))
    return g / g.sum()


def reciprocam(model, model_name, last_block_model, loaders, device="cpu"):
    # Define the 3x3 Gaussian weights
    gaussian_weights = generate_gaussian_mask(3, sigma=1.0)

    # Assuming the tokens (excluding class token) form a 14x14 grid
    grid_size = 14

    # Initialize the spatial masks
    N = 196  # Number of spatial masks (14x14 grid)
    T = 197  # Number of tokens (including the class token)
    M = np.zeros((N, T))

    # Fill the spatial masks with Gaussian weights (excluding the class token)
    for n in range(N):
        # Convert n to a spatial position (x, y) in the 2D grid of patches
        x = n // grid_size  # Row index
        y = n % grid_size  # Column index

        # Ensure the 3x3 area fits within the grid boundaries
        for dx in range(-1, 2):  # -1, 0, 1
            for dy in range(-1, 2):  # -1, 0, 1
                nx, ny = x + dx, y + dy
                if 0 <= nx < grid_size and 0 <= ny < grid_size:
                    idx = 1 + nx * grid_size + ny  # +1 to skip the class token
                    weight = gaussian_weights[
                        dx + 1, dy + 1
                    ]  # Offset by +1 to get correct weight
                    M[n, idx] = weight

    # Ensure the class token is not masked out
    for n in range(N):
        M[n, 0] = 1

    original_images_tensor = []

    # Get the node names for the layers we want to extract
    nodes = [f"encoder.layers.encoder_layer_11"]

    # Create an Extract model that only computes the desired intermediate activations
    model_ext = Inspect(model, layer=nodes)

    # Get the intermediate activations
    all_activations = {}
    all_targets = []

    saliency_maps = []

    already_exists = False

    for loader in loaders:
        if os.path.exists(f"./outputs/{model_name}/saliency_maps.safetensors"):
            already_exists = True
            print(f"Saliency Maps already exist...")
            continue

        with torch.no_grad():
            for images, targets in tqdm(loader):
                # Original image
                images_orig = images.to(device)
                all_targets.append(targets)

                for img in images:
                    original_images_tensor.append(img)

                # Run original image through the model
                probs, activations = model_ext(images_orig)

                activations = activations[0].cpu().numpy()

                # Initialize the new feature maps array
                b = activations.shape[0]  # Batch size
                T = activations.shape[1]  # Number of tokens (197)
                D = activations.shape[2]  # Hidden dimension (768)

                N = 196  # Number of spatial masks
                new_feature_maps = np.zeros((b, N, T, D))

                # Generate new feature maps via element-wise multiplication (Hadamard product)
                for n in range(N):
                    # Expand dimensions of mask to match feature map dimensions for element-wise multiplication
                    mask_expanded = M[n][
                        np.newaxis, :, np.newaxis
                    ]  # Shape: (1, 197, 1)

                    # Perform element-wise multiplication
                    new_feature_maps[:, n, :, :] = (
                        activations * mask_expanded
                    )  # Shape: (32, 197, 768)

                # Initialize a list to store the prediction scores
                scores = []

                # Process each new feature map and get the prediction scores
                for n in range(new_feature_maps.shape[1]):
                    # Extract the nth feature map (shape: (32, 197, 768))
                    feature_map_n = new_feature_maps[:, n, :, :]

                    # Convert the feature map to a PyTorch tensor if it isn't already
                    feature_map_n = torch.tensor(feature_map_n).float().to(device)

                    # Get the prediction scores for the specified class
                    with torch.no_grad():
                        output = last_block_model(feature_map_n)

                    # We are interested in the prediction score for the class with the highest probability
                    score = []
                    for i, out in enumerate(output):
                        score += [
                            F.softmax(out, dim=-1)[probs.argmax(dim=1)[i]].cpu().numpy()
                        ]

                    score = torch.tensor(np.array(score))

                    # Append the score to the list
                    scores.append(score.cpu().numpy())

                scores = np.array(scores).reshape(14, 14, -1)  # Shape: (14, 14, 32)

                # Normalize scores for better visualization
                scores = (scores - scores.min()) / (scores.max() - scores.min())

                # Generate saliency maps for each image in the batch
                batch_size = scores.shape[-1]

                for i in range(batch_size):
                    saliency_map = scores[:, :, i]
                    saliency_maps.append(saliency_map)

    if not already_exists:
        save_file(
            {"saliencymaps": torch.tensor(np.array(saliency_maps))},
            f"outputs/{model_name}/saliency_maps.safetensors",
        )

    return f"outputs/{model_name}/saliency_maps.safetensors"
