import torch
from safetensors.torch import save_file
from src.utils.utils import load_safetensor

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import numpy as np
import pandas as pd

import os

def compute_pca(all_activations, model_name):
    if os.path.exists(f"outputs/{model_name}/pca_results.safetensors"):
        pca_results = load_safetensor(f"outputs/{model_name}/pca_results.safetensors")

        print("PCA and TSNE results already exist. Skipping...")
    else:
        pca_results = {}

        layer_names = list(all_activations.keys())

        for layer_name in layer_names:
            # fit PCA on the original activations
            pca = PCA(n_components=50)
            pca_results[layer_name] = pca.fit_transform(
                all_activations[layer_name].numpy()
            )

        # save the pca results
        # convert the pca results to safetensors
        pca_results = {
            layer_name: torch.tensor(pca_result)
            for layer_name, pca_result in pca_results.items()
        }

        save_file(pca_results, f"outputs/{model_name}/pca_results.safetensors")

    return f"outputs/{model_name}/pca_results.safetensors"

def knn(pca_results, selected_layer, num_classes, seed=1):
    MAX_CLUSTERS = 10
    optimal_clusters = {}
    kmeans_results = {}

    silhouette_scores = []
    for n_clusters in range(num_classes + 1, MAX_CLUSTERS + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed)
        kmeans.fit(pca_results[selected_layer])
        score = silhouette_score(pca_results[selected_layer], kmeans.labels_)
        silhouette_scores.append((n_clusters, score))

    # Find the number of clusters with the highest silhouette score
    optimal_n_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
    optimal_clusters[selected_layer] = optimal_n_clusters

    # Fit KMeans with the optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=seed)
    kmeans.fit(pca_results[selected_layer])
    kmeans_results[selected_layer] = kmeans.labels_

    n_cluster = optimal_clusters[selected_layer]

    return kmeans_results, n_cluster

def select_cluster(n_cluster, kmeans_results, selected_layer, probabilities, target_labels):
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

    cluster_ids = df["total_score"].nlargest(1).index

    return cluster_ids