import torch
import torch.nn.functional as F
import numpy as np
import random
import os

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def normalize_ui_adj(indices, values, n_users, n_items, w=-0.5):
    rows, cols = indices
    deg_user = torch.zeros(n_users, device=values.device).scatter_add_(0, rows, values)
    deg_item = torch.zeros(n_items, device=values.device).scatter_add_(0, cols, values)
    deg_user_inv = deg_user.pow(w).nan_to_num_(0.0)
    deg_item_inv = deg_item.pow(-1-w).nan_to_num_(0.0)
    norm_values = deg_user_inv[rows] * values * deg_item_inv[cols]
    return torch.sparse_coo_tensor(indices, norm_values, size=(n_users, n_items), device=values.device).coalesce()

def info_nce(anchor, positive, temperature=0.2):
    anchor = F.normalize(anchor, dim=-1)
    positive = F.normalize(positive, dim=-1)
    logits = anchor @ positive.T
    labels = torch.arange(anchor.size(0), device=anchor.device)
    return torch.nn.functional.cross_entropy(logits / temperature, labels)

def calculate_recall(pred_items, gt_items):
    hits = len(set(pred_items) & gt_items)
    return hits / len(gt_items) if gt_items else 0.0

def calculate_ndcg(pred_items, gt_items, k):
    rel = [1 if item in gt_items else 0 for item in pred_items[:k]]
    dcg = sum((2 ** r - 1) / np.log2(idx + 2) for idx, r in enumerate(rel))
    idcg = sum((2 ** 1 - 1) / np.log2(idx + 2) for idx in range(min(len(gt_items), k)))
    return dcg / idcg if idcg > 0 else 0.0

def evaluate(scores, gts, top_k):
    results = {
        'recall': {k: [] for k in top_k},
        'ndcg': {k: [] for k in top_k}
    }

    for i in range(len(scores)):
        gt_items = gts[i]
        _, pred_items_all = torch.topk(scores[i], k=max(top_k))
        pred_items_all = pred_items_all.cpu().tolist()

        for k in top_k:
            pred_items = pred_items_all[:k]
            results['recall'][k].append(calculate_recall(pred_items, gt_items))
            results['ndcg'][k].append(calculate_ndcg(pred_items, gt_items, k))

    avg_results = {
        'recall': {k: np.mean(v) for k, v in results['recall'].items()},
        'ndcg': {k: np.mean(v) for k, v in results['ndcg'].items()}
    }

    return avg_results

def evaluate_group(scores, gts, top_k, user_basket_cnt):
    # define the group intervals and their names
    group_bins = [0, 2, 9, 16, float('inf')]
    group_names = ['<=2', '<=9', '<=16', '>16']
    group_indices = {name: [] for name in group_names}

    # assign users to the different groups
    for uid, cnt in user_basket_cnt.items():
        for i in range(len(group_bins) - 1):
            if group_bins[i] < cnt <= group_bins[i+1]:
                group_indices[group_names[i]].append(uid)
                break

    # initialize the result dict for each group
    group_results = {
        name: {
            'recall': {k: [] for k in top_k},
            'ndcg': {k: [] for k in top_k}
        }
        for name in group_names
    }

    # iterate over each user and record their metrics by group
    for i in range(len(scores)):
        gt_items = gts[i]
        _, pred_items_all = torch.topk(scores[i], k=max(top_k))
        pred_items_all = pred_items_all.cpu().tolist()

        # find which group the current user belongs to
        user_cnt = user_basket_cnt[i]
        group_name = None
        for j in range(len(group_bins) - 1):
            if group_bins[j] < user_cnt <= group_bins[j+1]:
                group_name = group_names[j]
                break

        for k in top_k:
            pred_items = pred_items_all[:k]
            group_results[group_name]['recall'][k].append(calculate_recall(pred_items, gt_items))
            group_results[group_name]['ndcg'][k].append(calculate_ndcg(pred_items, gt_items, k))

    # compute the average results for each group
    group_avg_results = {}
    for name in group_names:
        group_avg_results[name] = {
            'recall': {k: np.mean(v) if v else 0.0 for k, v in group_results[name]['recall'].items()},
            'ndcg': {k: np.mean(v) if v else 0.0 for k, v in group_results[name]['ndcg'].items()}
        }

    return group_avg_results