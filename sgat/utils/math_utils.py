import numpy as np

import torch

def score_weights(scores, split, upsampled_weight):
    """If sample has a score south of a random split point, it is more likely to be sampled in the batch."""
    weights = np.zeros(len(scores))
    for idx, score in enumerate(scores):
        if score < split:
            weight = upsampled_weight
        else:
            weight = 1 - upsampled_weight
        weights[idx] = weight
    return weights

def exp_weighted_mse(output, target, exp_loc, exp_scale, eps=1e-3):
    """Custom loss function assigning greater weight to errors at the top of the ranked list."""
    weight = torch.clamp((torch.exp(-(target - exp_loc) / exp_scale) / exp_scale), min=0.0, max=1)
    loss = torch.mean(weight * (output - target) ** 2) + eps
    return loss
