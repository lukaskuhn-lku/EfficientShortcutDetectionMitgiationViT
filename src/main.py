import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, ConcatDataset

from torchvision.models import vit_b_16
import torchvision.transforms as transforms
from torchvision import datasets, transforms

from tqdm import tqdm
import numpy as np
from surgeon_pytorch import Extract
import os

from utils.captioning import caption_images, summarize_cluster
from utils.models.vit_last_block import ViTLastBlock
from utils.utils import (
    extract_number,
    get_n_closest_points,
    load_safetensor,
    load_subset_by_target,
)
from utils.inference import inference_for_loader
from utils.patches import get_patches_in_cluster, reverse_transform
from utils.surgeon import get_predictions
from utils.reciprocam import reciprocam
from utils.clustering import compute_pca, knn, select_cluster

from sklearn.cluster import KMeans

import pandas as pd
import replicate
from utils.visualization import save_patches


def main():
    SEED = 1
    MODEL_NAME = f"vit_isic_{SEED}"
    NUM_CLASSES = 2

    # Set the seed for reproducibility
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Define the mean and standard deviation of the dataset
    mean_ds = [0.485, 0.456, 0.406]
    std_ds = [0.229, 0.224, 0.225]

    # Define the transformations to apply to the images
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_ds, std=std_ds),
        ]
    )

    # Create the ImageFolder dataset
    train_dataset_wo_patches = datasets.ImageFolder(
        root="./data/isic/val_wo_patches/", transform=transform
    )
    train_dataset_w_patches = datasets.ImageFolder(
        root="./data/isic/val_w_patches/", transform=transform
    )

    concatenated_dataset = ConcatDataset(
        [train_dataset_wo_patches, train_dataset_w_patches]
    )

    isic_loader = torch.utils.data.DataLoader(
        concatenated_dataset, batch_size=32, shuffle=False
    )

    loaders = [isic_loader]
    loader_names = ["isic_loader"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the pre-trained ViT model
    model = vit_b_16(weights="DEFAULT")

    # Modify the final layer for the dataset
    model.heads.head = nn.Linear(model.heads.head.in_features, NUM_CLASSES)
    model.to(device)
    model.eval()

    # load the pre-trained model
    model.load_state_dict(torch.load(f"models/{MODEL_NAME}.pth", map_location="cuda"))

    # Create an instance of the extracted model
    last_block_model = ViTLastBlock(model)
    last_block_model.to(device)
    last_block_model.eval()

    # Extract Key weights from the last block of the model
    self_attention_layer = model.encoder.layers[-1].self_attention

    # Extract the dimensions
    embed_dim = 768

    # Extract the weight matrix for the keys (Wk) from the combined weight matrix
    W_K = self_attention_layer.in_proj_weight[embed_dim : 2 * embed_dim, :]

    conv_out_model = Extract(model, node_out="encoder.dropout")

    # Get layer names
    layer_names = [f"encoder.layers.encoder_layer_{i}" for i in range(12)]
    layer_names.sort(key=extract_number)

    # Get the predictions for the model
    prediction_results = [
        get_predictions(
            loader=loader,
            loader_name=loader_names[i],
            model=model,
            model_name=MODEL_NAME,
            layer_names=layer_names,
            conv_out_model=conv_out_model,
            W_K=W_K,
            device=device,
        )
        for i, loader in enumerate(loaders)
    ]

    saliency_map_path = reciprocam(
        model=model,
        model_name=MODEL_NAME,
        last_block_model=last_block_model,
        loaders=loaders,
        device=device,
    )

    activations = [
        load_safetensor(prediction_result[0])
        for prediction_result in prediction_results
    ]
    saliency_maps = load_safetensor(saliency_map_path)

    # concatenate each layer's activations
    all_activations = {}
    for layer_name in layer_names:
        all_activations[layer_name] = torch.cat(
            [act[layer_name] for act in activations], dim=0
        )

    # load the targets
    targets = [
        load_safetensor(prediction_result[1])
        for prediction_result in prediction_results
    ]
    target_labels = torch.cat([target["targets"] for target in targets], dim=0)

    pca_path = compute_pca(all_activations, MODEL_NAME)
    pca_results = load_safetensor(pca_path)

    selected_layer = "encoder.layers.encoder_layer_11"
    kmeans_results, n_cluster = knn(pca_results, selected_layer, NUM_CLASSES)

    logits = [
        load_safetensor(prediction_result[2])
        for prediction_result in prediction_results
    ]
    probabilities = [F.softmax(logit["logits"], dim=-1) for logit in logits]

    cluster_ids = select_cluster(n_cluster, kmeans_results, selected_layer, probabilities, target_labels)

    n = 10 # Number of closest points to extract
    n_closest_points_indices = get_n_closest_points(
        pca_results=pca_results,
        kmeans_results=kmeans_results,
        n=n,
        n_cluster=n_cluster,
        seed=SEED,
    )

    keys = load_safetensor(f"outputs/{MODEL_NAME}/isic_loader_keys.safetensors")["keys"]

    all_top_indices, all_image_indices, all_patches, all_overlays = (
        get_patches_in_cluster(
            images=keys,
            saliency_maps=saliency_maps,
            kmeans_results=kmeans_results,
            n_closest_points_indices=n_closest_points_indices,
            layer_name=selected_layer,
            n_cluster=n_cluster,
            concatenated_dataset=concatenated_dataset,
            device=device,
        )
    )

    patch_files = save_patches(n_cluster, all_patches, all_overlays, all_image_indices, concatenated_dataset, MODEL_NAME, reverse_transform)


    caption_model = (
            "yorickvp/llava-13b:b5f6212d032508382d61ff00469ddda3e32fd8a0e75dc39d8a4191bb742157fb",
            "llava-13b",
    )

    replicate_api = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])
    cluster_descriptions = caption_images(replicate_api, caption_model, patch_files, n_cluster, MODEL_NAME)

    # summarizes the cluster via LLM and outputs the results to a file
    summarize_cluster(replicate_api, cluster_descriptions, MODEL_NAME)

    top_indices = all_top_indices[cluster_ids.values[0]]
    image_indices = all_image_indices[cluster_ids.values[0]]

    combined_keys = []

    for idx, image in enumerate(image_indices):
        for i, coord in enumerate(top_indices[idx]):
            key_idx = coord[0] * 14 + coord[1]
            combined_keys.append(keys[image][key_idx])

    combined_keys = np.array(combined_keys).reshape(n * 10, -1)
    combined_keys = combined_keys / np.linalg.norm(combined_keys, axis=1)[:, np.newaxis]

    # Define the transformations to apply to the images
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),  # Resize the images to a fixed size
            transforms.ToTensor(),  # Convert the images to tensors
            transforms.Normalize(mean=mean_ds, std=std_ds),  # Normalize the images
        ]
    )

    # Create the ImageFolder dataset
    test_dataset_wo_patches = datasets.ImageFolder(
        root="./data/isic/test_wo_patches/", transform=transform
    )
    test_dataset_w_patches = datasets.ImageFolder(
        root="./data/isic/test_w_patches/", transform=transform
    )

    patches_malignant = load_subset_by_target(test_dataset_w_patches, 1)
    patches_benign = load_subset_by_target(test_dataset_w_patches, 0)

    no_patches_malignant = load_subset_by_target(test_dataset_wo_patches, 1)
    no_patches_benign = load_subset_by_target(test_dataset_wo_patches, 0)

    patches_malignant_loader = DataLoader(
        patches_malignant, batch_size=1, shuffle=False
    )
    patches_benign_loader = DataLoader(patches_benign, batch_size=1, shuffle=False)

    no_patches_malignant_loader = DataLoader(
        no_patches_malignant, batch_size=1, shuffle=False
    )
    no_patches_benign_loader = DataLoader(
        no_patches_benign, batch_size=1, shuffle=False
    )

    (
        correct_mal_w_patch,
        correct_ab_mal_w_patch,
        total_mal_w_patch,
        _,
        _,
    ) = inference_for_loader(
        patches_malignant_loader, model, W_K, combined_keys, device
    )
    print(correct_mal_w_patch, correct_ab_mal_w_patch, total_mal_w_patch)
    (
        correct_ben_w_patch,
        correct_ab_ben_w_patch,
        total_ben_w_patch,
        _,
        _,
    ) = inference_for_loader(patches_benign_loader, model, W_K, combined_keys, device)
    print(correct_ben_w_patch, correct_ab_ben_w_patch, total_ben_w_patch)

    (
        correct_mal_wo_patch,
        correct_ab_mal_wo_patch,
        total_mal_wo_patch,
        _,
        _,
    ) = inference_for_loader(
        no_patches_malignant_loader, model, W_K, combined_keys, device
    )
    print(correct_mal_wo_patch, correct_ab_mal_wo_patch, total_mal_wo_patch)
    (
        correct_ben_wo_patch,
        correct_ab_ben_wo_patch,
        total_ben_wo_patch,
        _,
        _,
    ) = inference_for_loader(
        no_patches_benign_loader, model, W_K, combined_keys, device
    )
    print(correct_ben_wo_patch, correct_ab_ben_wo_patch, total_ben_wo_patch)
