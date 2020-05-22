from itertools import permutations

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import adjusted_rand_score


def combinatorical_cce(y_pred, y_true):
    loss = 0
    for y_pred_item, y_true_item in zip(y_pred, y_true):
        lowest_cce = None
        for class_mapping in list(permutations(range(y_pred_item.shape[1]))):  # find best permutation
            current_cce = F.cross_entropy(y_pred_item[:, class_mapping], y_true_item)
            if lowest_cce is None or current_cce < lowest_cce:
                lowest_cce = current_cce
        loss += lowest_cce
    return loss / y_pred.shape[0]


def combinatorical_accuracy(y_pred, y_true):
    acc = 0
    for y_pred_item, y_true_item in zip(y_pred, y_true):
        highest_acc = None
        for class_mapping in list(permutations(range(y_pred_item.shape[1]))):  # find best permutation
            current_acc = torch.mean((torch.argmax(y_pred_item[:, class_mapping], dim=1) == y_true_item).float()).item()
            if highest_acc is None or current_acc > highest_acc:
                highest_acc = current_acc
        acc += highest_acc
    return acc


def ari_score(y_pred, y_true):
    ari = []
    for y_pred_item, y_true_item in zip(y_pred, y_true):
        y_pred_item = torch.argmax(y_pred_item, dim=1).data.cpu().numpy()
        ari_item = adjusted_rand_score(y_pred_item, y_true_item.cpu().numpy())
        ari.append(ari_item)
    ari = np.mean(ari)
    return ari
