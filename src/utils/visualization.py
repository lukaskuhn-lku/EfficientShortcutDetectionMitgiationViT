import os
import matplotlib.pyplot as plt

def save_patches(n_cluster, all_patches, all_overlays, all_image_indices, concatenated_dataset, model_name, reverse_transform):
    patch_files = []

    for cluster in range(n_cluster):
        os.makedirs(f"outputs/{model_name}/patches/{cluster}", exist_ok=True)

        patch_f = []

        for idx, patch in enumerate(all_patches[cluster]):
            # save image
            plt.imshow(patch)
            plt.axis("off")
            plt.savefig(
                f"outputs/{model_name}/patches/{cluster}/{idx}.png",
                bbox_inches="tight",
                pad_inches=0,
            )
            patch_f.append(f"outputs/{model_name}/patches/{cluster}/{idx}.png")

        os.makedirs(f"outputs/{model_name}/overlays/{cluster}", exist_ok=True)

        for idx, patch in enumerate(all_overlays[cluster]):
            # save image
            plt.imshow(patch)
            plt.axis("off")
            plt.savefig(
                f"outputs/{model_name}/overlays/{cluster}/{idx}.png",
                bbox_inches="tight",
                pad_inches=0,
            )

        patch_files.append(patch_f)

        for idx, orig_idx in enumerate(all_image_indices[cluster]):
            # save image
            plt.imshow(reverse_transform(concatenated_dataset[orig_idx][0]))
            plt.axis("off")
            plt.savefig(
                f"outputs/{model_name}/overlays/{cluster}/{idx}_original.png",
                bbox_inches="tight",
                pad_inches=0,
            )

    return patch_files