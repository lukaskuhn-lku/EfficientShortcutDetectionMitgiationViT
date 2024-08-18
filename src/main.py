import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import re

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, ConcatDataset

from torchvision.models import vit_b_16
import torchvision.transforms as transforms
from torchvision import datasets, transforms

from tqdm import tqdm
import numpy as np
from surgeon_pytorch import Inspect, get_nodes, get_layers, Extract
import os
import gc

from safetensors import safe_open
from safetensors.torch import save_file

from sklearn.decomposition import PCA
import time

from src.utils.models.vit_last_block import ViTLastBlock
from src.utils.utils import (
    extract_number,
    get_n_closest_points,
    load_safetensor,
    load_subset_by_target,
)
from utils.inference import inference_for_loader
from utils.patches import get_patches_in_cluster, reverse_transform
from utils.surgeon import get_predictions
from utils.reciprocam import reciprocam

from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import pandas as pd
import replicate


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
    kdim = self_attention_layer.kdim

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

    # PCA
    if (
        os.path.exists(f"outputs/{MODEL_NAME}/pca_results.safetensors")
        and os.path.exists(f"outputs/{MODEL_NAME}/pca_3d_results.safetensors")
        and os.path.exists(f"outputs/{MODEL_NAME}_{SEED}/tsne_3d_results.safetensors")
    ):
        pca_results = load_safetensor(f"outputs/{MODEL_NAME}/pca_results.safetensors")
        pca_3d_results = load_safetensor(
            f"outputs/{MODEL_NAME}/pca_3d_results.safetensors"
        )
        tsne_results = load_safetensor(
            f"outputs/{MODEL_NAME}/tsne_3d_results.safetensors"
        )

        print("PCA and TSNE results already exist. Skipping...")
    else:
        pca_results = {}
        pca_3d_results = {}
        tsne_3d_results = {}

        layer_names = list(all_activations.keys())

        for layer_name in layer_names:
            # fit PCA on the original activations
            pca = PCA(n_components=50)
            pca_3d = PCA(n_components=3)
            pca_results[layer_name] = pca.fit_transform(
                all_activations[layer_name].numpy()
            )
            pca_3d_results[layer_name] = pca_3d.fit_transform(
                all_activations[layer_name].numpy()
            )
            tsne_3d_results[layer_name] = TSNE(n_components=3).fit_transform(
                all_activations[layer_name].numpy()
            )

        # save the pca results
        # convert the pca results to safetensors
        pca_3d_results = {
            layer_name: torch.tensor(pca_result)
            for layer_name, pca_result in pca_3d_results.items()
        }
        tsne_results = {
            layer_name: torch.tensor(tsne_results)
            for layer_name, tsne_results in tsne_3d_results.items()
        }
        pca_results = {
            layer_name: torch.tensor(pca_result)
            for layer_name, pca_result in pca_results.items()
        }

        save_file(pca_3d_results, f"outputs/{MODEL_NAME}/pca_3d_results.safetensors")
        save_file(tsne_results, f"outputs/{MODEL_NAME}/tsne_3d_results.safetensors")
        save_file(pca_results, f"outputs/{MODEL_NAME}/pca_results.safetensors")

    MAX_CLUSTERS = 10
    optimal_clusters = {}
    kmeans_results = {}
    cluster_to_take = pca_results

    for layer_name in layer_names:
        silhouette_scores = []
        for n_clusters in range(NUM_CLASSES + 1, MAX_CLUSTERS + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=SEED)
            kmeans.fit(cluster_to_take[layer_name])
            score = silhouette_score(cluster_to_take[layer_name], kmeans.labels_)
            silhouette_scores.append((n_clusters, score))

        # Find the number of clusters with the highest silhouette score
        optimal_n_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
        optimal_clusters[layer_name] = optimal_n_clusters

        # Fit KMeans with the optimal number of clusters
        kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=SEED)
        kmeans.fit(cluster_to_take[layer_name])
        kmeans_results[layer_name] = kmeans.labels_

    n_cluster = optimal_clusters["encoder.layers.encoder_layer_11"]
    print(f"Optimal number of clusters: {n_cluster}")

    logits = [
        load_safetensor(prediction_result[2])
        for prediction_result in prediction_results
    ]
    probabilities = [F.softmax(logit["logits"], dim=-1) for logit in logits]

    selected_layer = "encoder.layers.encoder_layer_11"

    scores = []

    for cluster_idx in range(n_cluster):
        cluster_indices = torch.where(
            torch.tensor(kmeans_results[selected_layer]) == cluster_idx
        )[0]
        cluster_probabilities = probabilities[0][cluster_indices]
        cluster_targets = target_labels[cluster_indices]

        # count the number of 1 and 0s in the target labels
        n_ones = torch.sum(cluster_targets)
        n_zeros = len(cluster_targets) - n_ones

        dominant = 0 if n_zeros > n_ones else 1
        non_dominant = 1 - dominant

        cluster_probabilities_where_dominant = cluster_probabilities[
            cluster_targets == dominant
        ]
        cluster_probabilities_where_non_dominant = cluster_probabilities[
            cluster_targets == non_dominant
        ]

        # calculate brier score for the dominant class
        brier_score_dominant = torch.mean(
            (cluster_probabilities_where_dominant[:, dominant] - 1) ** 2
        )
        brier_score_non_dominant = torch.mean(
            cluster_probabilities_where_non_dominant[:, dominant] ** 2
        )

        heterogeneity_score = (
            n_zeros / len(cluster_indices)
            if n_zeros > n_ones
            else n_ones / len(cluster_indices)
        )

        print(f"Cluster {cluster_idx}")
        print(f"Brier Score Dominant: {np.exp(-brier_score_dominant.item())}")
        print(f"Brier Score Non-Dominant: {1-np.exp(-brier_score_non_dominant.item())}")
        print(f"Heterogeneity Score: {heterogeneity_score}")

        brier_score_non_dominant = (
            brier_score_non_dominant
            if brier_score_non_dominant > 0
            else torch.tensor(0)
        )

        scores.append(
            {
                "cluster_idx": cluster_idx,
                "brier_score_dominant": np.exp(-brier_score_dominant.item()),
                "brier_score_non_dominant": 1
                - np.exp(-brier_score_non_dominant.item()),
                "n_samples": len(cluster_indices),
                "heterogenity_score": heterogeneity_score.item(),
                "n_zeros": n_zeros.item(),
                "n_ones": n_ones.item(),
            }
        )

    df = pd.DataFrame(scores)

    df["total_score"] = (
        df["brier_score_dominant"]
        + df["brier_score_non_dominant"]
        + df["heterogenity_score"]
    )

    cluster_ids = df["total_score"].nlargest(n_clusters).index

    print(df.head())

    # Example usage
    n = 10  # Number of most distant points to retrieve
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

    patch_files = []

    for cluster in range(n_cluster):
        os.makedirs(f"outputs/{MODEL_NAME}/patches/{cluster}", exist_ok=True)

        patch_f = []

        for idx, patch in enumerate(all_patches[cluster]):
            # save image
            plt.imshow(patch)
            plt.axis("off")
            plt.savefig(
                f"outputs/{MODEL_NAME}/patches/{cluster}/{idx}.png",
                bbox_inches="tight",
                pad_inches=0,
            )
            patch_f.append(f"outputs/{MODEL_NAME}/patches/{cluster}/{idx}.png")

        os.makedirs(f"outputs/{MODEL_NAME}/overlays/{cluster}", exist_ok=True)

        for idx, patch in enumerate(all_overlays[cluster]):
            # save image
            plt.imshow(patch)
            plt.axis("off")
            plt.savefig(
                f"outputs/{MODEL_NAME}/overlays/{cluster}/{idx}.png",
                bbox_inches="tight",
                pad_inches=0,
            )

        patch_files.append(patch_f)

        for idx, orig_idx in enumerate(all_image_indices[cluster]):
            # save image
            plt.imshow(reverse_transform(concatenated_dataset[orig_idx][0]))
            plt.axis("off")
            plt.savefig(
                f"outputs/{MODEL_NAME}/overlays/{cluster}/{idx}_original.png",
                bbox_inches="tight",
                pad_inches=0,
            )

    os.environ["REPLICATE_API_TOKEN"] = ""
    api = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])

    caption_models = [
        (
            "yorickvp/llava-13b:b5f6212d032508382d61ff00469ddda3e32fd8a0e75dc39d8a4191bb742157fb",
            "llava-13b",
        )
    ]

    for cap_mod in caption_models:
        cluster_descriptions = []

        for cluster in range(n_cluster):
            descriptions = []
            for idx, image in tqdm(enumerate(patch_files[cluster])):
                output = api.run(
                    cap_mod[0],
                    input={
                        "image": open(image, "rb"),
                        "prompt": "What is in this picture? Describe in a few words.",
                    },
                )
                descriptions.append("".join(output))

            cluster_descriptions.append(descriptions)

            os.makedirs(
                f"outputs/{MODEL_NAME}_{SEED}/descriptions/{cluster}", exist_ok=True
            )
            # write descriptions to file
            with open(
                f"outputs/{MODEL_NAME}_{SEED}/descriptions/{cluster}/{cap_mod[1]}_descriptions.txt",
                "w",
            ) as f:
                f.write("\n".join(descriptions))

    for cluster in range(n_cluster):
        input = {
            "prompt": f"I extracted patches from images in my dataset where my model seems to focus on the most. I let an LLM caption these images for you. I am searching for potential shortcuts in the dataset. Can you identify one or more possible shortcuts in this dataset? Describe it in one sentence (only!) and pick the most significant. No other explanations are needed. If there isn't any shortcut obvious to you then just say so. Descriptions: \n"
            + "".join(cluster_descriptions[cluster]),
            "prompt_template": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
        }

        output = api.run("meta/meta-llama-3-70b-instruct", input=input)

        text = "".join(output)

        with open(
            f"outputs/{MODEL_NAME}_{SEED}/descriptions/{cluster}/llama70b_shortcut.txt",
            "w",
        ) as f:
            f.write(text)

    top_indices = all_top_indices[cluster_ids.values[0]]
    image_indices = all_image_indices[cluster_ids.values[0]]
    patches = all_patches[cluster_ids.values[0]]

    combined_keys = []

    for idx, image in enumerate(image_indices):
        for i, coord in enumerate(top_indices[idx]):
            key_idx = coord[0] * 14 + coord[1]
            combined_keys.append(keys[image][key_idx])

    combined_keys = np.array(combined_keys)
    combined_keys = combined_keys.reshape(n * 10, -1)
    combined_keys = combined_keys / np.linalg.norm(combined_keys, axis=1)[:, np.newaxis]

    mean_ds = [0.485, 0.456, 0.406]
    std_ds = [0.229, 0.224, 0.225]

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

    test_concatenated_dataset = ConcatDataset(
        [test_dataset_wo_patches, test_dataset_w_patches]
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
