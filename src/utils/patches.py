import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms


# Helper functions
def create_heatmap(similarity_means, token_indices, grid_size=14):
    heatmap = np.zeros((grid_size, grid_size))
    for idx, mean in zip(token_indices, similarity_means):
        row = idx // grid_size
        col = idx % grid_size
        heatmap[row, col] = mean
    return heatmap


def resize_heatmap(heatmap, image_size=224):
    resized_heatmap = cv2.resize(
        heatmap, (image_size, image_size), interpolation=cv2.INTER_CUBIC
    )
    return resized_heatmap


def overlay_heatmap_on_image(image, heatmap, alpha=0.8, colormap=cv2.COLORMAP_JET):
    # Normalize the heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap = np.uint8(255 * heatmap)

    # Apply colormap
    heatmap = cv2.applyColorMap(heatmap, colormap)

    # Blend heatmap with the original image
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

    return overlay


def overlay_saliency_on_image(original_image, saliency_map):
    # Normalize the saliency map
    saliency_map = (saliency_map - saliency_map.min()) / (
        saliency_map.max() - saliency_map.min()
    )
    saliency_map = np.uint8(255 * saliency_map)
    saliency_map = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
    saliency_map = cv2.cvtColor(
        saliency_map, cv2.COLOR_BGR2RGB
    )  # Convert color map from BGR to RGB

    # Ensure both images are the same type
    original_image = original_image.astype(np.float32)
    saliency_map = saliency_map.astype(np.float32)

    overlay = cv2.addWeighted(original_image, 0.3, saliency_map, 0.7, 0)
    overlay = np.uint8(overlay)  # Convert back to uint8
    return overlay


def euclidean_distance(v1, v2):
    return torch.sqrt(torch.sum((v1 - v2) ** 2, dim=-1))


def map_indices_to_coordinates(indices, grid_size=14, image_size=224, patch_size=16):
    """
    Map 14x14 grid indices to 224x224 image coordinates.

    Args:
        indices (list of tuples): List of (row, col) indices in the 14x14 grid.
        grid_size (int): Size of the grid (default is 14).
        image_size (int): Size of the image (default is 224).
        patch_size (int): Size of each patch (default is 16).

    Returns:
        list of tuples: List of (x, y) coordinates in the 224x224 image.
    """
    coordinates = []
    for row, col in indices:
        x = col * patch_size
        y = row * patch_size
        coordinates.append((x, y))
    return coordinates


def reverse_transform(
    image_tensor, mean_ds=[0.485, 0.456, 0.406], std_ds=[0.229, 0.224, 0.225]
):
    # Step 1: Un-normalize the images
    mean = torch.tensor(mean_ds).reshape(1, 3, 1, 1)
    std = torch.tensor(std_ds).reshape(1, 3, 1, 1)
    image_tensor = image_tensor * std + mean

    # Step 2: Convert tensors back to images
    to_pil = transforms.ToPILImage()
    image = to_pil(image_tensor.squeeze())  # Squeeze to remove batch dimension

    return image


def get_patches_in_cluster(
    images,
    saliency_maps,
    kmeans_results,
    n_closest_points_indices,
    layer_name,
    n_cluster,
    concatenated_dataset,
    device="cpu",
):
    # HyperParameters
    n_most_active = 10

    # Group images by cluster
    images_by_cluster = [
        images[kmeans_results[layer_name] == i] for i in range(n_cluster)
    ]

    all_top_indices = []
    all_image_indices = []
    all_patches = []
    all_overlays = []

    # Step 2 Loop over the clusters
    for cluster in range(n_cluster):
        top_indices = []
        image_indices = []
        patches = []
        overlays = []

        # Define the other clusters
        other_cluster_ids = [i for i in range(n_cluster) if i != cluster]
        other_cluster_keys = np.concatenate(
            [images_by_cluster[i] for i in other_cluster_ids], axis=0
        )
        other_cluster_keys = torch.tensor(other_cluster_keys).to(device)

        # Use the most distant points in the cluster
        for idx, index in enumerate(n_closest_points_indices[layer_name][cluster]):
            # Grab the saliency map for the current index

            image_indices.append(index)

            smap = saliency_maps[index]
            keys_for_target = images[index]

            cos_sim_means = []
            for active_key in keys_for_target:
                active_key = active_key.unsqueeze(0).to(device)

                # Compute Euclidean distance and invert it
                distances = euclidean_distance(
                    active_key, other_cluster_keys.view(-1, 64)
                ) 
                mean_distance = distances.mean().item()
                cos_sim_means.append(mean_distance)

            # Create the heatmap
            heatmap = np.array(cos_sim_means).reshape(
                14, 14
            )  # create_heatmap(cos_sim_means, range(196))  # Correct the range
            heatmap = cv2.normalize(heatmap, heatmap, 0, 1, cv2.NORM_MINMAX)

            resized_heatmap = resize_heatmap(heatmap)

            # smap = cv2.normalize(smap.numpy(), smap.numpy(), 0, 1, cv2.NORM_MINMAX)
            smap = smap.numpy()
            resized_smap = resize_heatmap(smap)

            # Combine the heatmaps
            heat = resized_heatmap * resized_smap

            # Get the original image and reverse transform it
            image = concatenated_dataset[index][0]
            image = reverse_transform(image)

            # Overlay the heatmap on the image
            overlay = overlay_saliency_on_image(np.array(image), heat)

            heat_normal = heatmap * smap  # 14x14 map

            heat_normal_flat = heat_normal.flatten()

            # Get indices of the top n_most_active values
            top_n_indices_flat = heat_normal_flat.argsort()[-n_most_active:][::-1]

            # Convert flat indices back to 2D indices
            top_n_indices_2d = [(idx // 14, idx % 14) for idx in top_n_indices_flat]

            top_indices.append(top_n_indices_2d)

            # Extract the patch around the highest activation
            highest_activation_idx = top_n_indices_2d[0]
            center_x, center_y = map_indices_to_coordinates([highest_activation_idx])[0]
            half_patch = 100 // 2
            x_start = max(center_x - half_patch, 0)
            x_end = min(center_x + half_patch, 224)
            y_start = max(center_y - half_patch, 0)
            y_end = min(center_y + half_patch, 224)

            patch = np.array(image)[y_start:y_end, x_start:x_end]
            patches.append(patch)

            overlays.append(overlay)

        all_top_indices.append(top_indices)
        all_image_indices.append(image_indices)
        all_patches.append(patches)
        all_overlays.append(overlays)

    return all_top_indices, all_image_indices, all_patches, all_overlays
