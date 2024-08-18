from surgeon_pytorch import Extract
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils.utils import add_surrounding_patches, get_mask_from_indices


def inference_for_loader(loader, model, W_K, combined_keys, device="cpu"):
    conv_out_model = Extract(model, node_out="encoder.dropout")
    conv_in_model = Extract(model, node_in="encoder.dropout", node_out="heads.head")

    correct_abilation = 0
    correct = 0
    total = 0
    ablated = 0
    mse = []
    mse_abilation = []
    correct_images = []

    for img, target in tqdm(loader):
        all_patches = conv_out_model(img.to(device))[0, 1:, :].detach().cpu().numpy()

        ln_1 = torch.tensor(all_patches.reshape(196, -1)).to(device)
        k = ln_1 @ W_K.T

        k = k.reshape(196, 12, 64)
        all_keys = k.mean(dim=1).detach().cpu()

        means = []
        for k in all_keys:
            k = k / np.linalg.norm(k)
            dist = cosine_similarity(
                np.array([k]), combined_keys
            )  # cosine_similarity(np.array([k]), combined_keys)
            means.append(dist.max().item())

        # set all the means to 0 where the probability is less than 0.75
        threshold = 0.80
        means = np.array(means)
        means[means < threshold] = 0

        indices = np.where(means > 0)[0]

        indices = np.array(add_surrounding_patches(indices))

        indices += 1

        if len(indices) > 0:
            ablated += 1

        mask = get_mask_from_indices(indices)

        x = img.to(device)
        encoder = conv_out_model(x)
        enc = encoder[:, mask, :]

        pred_abilation = F.softmax(conv_in_model(x, enc), dim=1).argmax().item()
        pred = F.softmax(model(x), dim=1).argmax().item()

        if pred == target.item():
            correct += 1

        if pred_abilation == target.item():
            correct_abilation += 1

        if pred != pred_abilation:
            correct_images.append(img)

        total += 1

    return correct, correct_abilation, total, correct_images, ablated
