import re
from safetensors import safe_open
import numpy as np
from sklearn.cluster import KMeans
import torch


def extract_number(layer_name):
    match = re.search(r"\d+", layer_name)
    return int(match.group()) if match else float("inf")


def load_safetensor(activation_path):
    activations = {}
    with safe_open(activation_path, framework="pt", device="cuda") as f:
        for k in f.keys():
            activations[k] = f.get_tensor(k).cpu()

    return activations


def get_n_closest_points(pca_results, kmeans_results, n, n_cluster, seed=42):
    closest_points_indices = {}

    for layer_name in pca_results.keys():
        centroids = (
            KMeans(n_clusters=n_cluster, random_state=seed)
            .fit(pca_results[layer_name])
            .cluster_centers_
        )
        closest_points_indices[layer_name] = []

        for i in range(n_cluster):
            cluster_points = pca_results[layer_name][kmeans_results[layer_name] == i]
            distances = np.linalg.norm(cluster_points - centroids[i], axis=1)
            closest_indices = np.argsort(distances)[:n]
            original_indices = np.where(kmeans_results[layer_name] == i)[0][
                closest_indices
            ]
            closest_points_indices[layer_name].append(original_indices)

    return closest_points_indices


def get_patch_from_image(image, top_left_coordinates, patch_size=16):
    x, y = top_left_coordinates
    patch = image[y : y + patch_size, x : x + patch_size, :]
    return patch


def get_surrounding_indices(index, grid_size):
    patch_size = 16
    num_patches = grid_size // patch_size
    row = index // num_patches
    col = index % num_patches

    surrounding_indices = []

    # Define the possible directions (top, bottom, left, right, and diagonals)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for dr, dc in directions:
        new_row = row + dr
        new_col = col + dc

        # Check if the new coordinates are within bounds
        if 0 <= new_row < num_patches and 0 <= new_col < num_patches:
            surrounding_index = new_row * num_patches + new_col
            surrounding_indices.append(surrounding_index)

    return surrounding_indices


def add_surrounding_patches(indices, grid_size=224):
    # Convert the list of indices to a set for efficient look-up
    indices_set = set(indices)

    new_indices = set(indices)

    for index in indices:
        surrounding_indices = get_surrounding_indices(index, grid_size)
        for surrounding_index in surrounding_indices:
            if surrounding_index not in indices_set:
                new_indices.add(surrounding_index)

    return list(new_indices)


def get_mask_from_indices(indices):
    mask = torch.ones(197, dtype=bool)
    mask[indices] = False
    return mask


def load_subset_by_target(ds, target_class):
    indices = [i for i, (_, target) in enumerate(ds) if target == target_class]

    subset = torch.utils.data.Subset(ds, indices)
    return subset
