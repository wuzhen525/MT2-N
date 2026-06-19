import torch
import pandas as pd
import numpy as np
from utils import normalize_ui_adj
from torch.utils.data import Dataset
import torch.nn.functional as F
from collections import defaultdict
import os

class TrainDataset(Dataset):
    def __init__(self, train_df, train_user_baskets, num_items):
        self.num_items = num_items
        self.user_baskets = train_user_baskets

        last_baskets = train_df.groupby('user_id')['basket_ord'].max().reset_index()
        last_basket_records = pd.merge(
            last_baskets,
            train_df,
            on=['user_id', 'basket_ord'],
            how='left'
        )
        self.user_last_basket = last_basket_records.groupby('user_id')['item_id'].apply(list).to_dict()

        all_user_baskets = train_df.groupby('user_id')['item_id'].apply(list).to_dict()
        self.user_side_items = {}
        for user_id in self.user_last_basket:
            all_items = all_user_baskets.get(user_id, [])
            last_items = set(self.user_last_basket[user_id])
            side_items = list(set(all_items) - last_items)
            self.user_side_items[user_id] = side_items

        self.user_ids = list(self.user_last_basket.keys())

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        pos_items = self.user_last_basket[user_id]

        user_tensor = torch.tensor([user_id] * len(pos_items), dtype=torch.long)
        pos_tensor = torch.tensor(pos_items, dtype=torch.long)

        return idx, user_tensor, pos_tensor, self.user_baskets[user_id]

def collate_fn(batch, device):
    idxs = [b[0] for b in batch]
    user = torch.cat([b[1] for b in batch], dim=0)
    pos = torch.cat([b[2] for b in batch], dim=0)
    user_baskets = [b[3] for b in batch]
    user_baskets_tensor = torch.stack(user_baskets, dim=0)

    sorted_idx = sorted(range(len(idxs)), key=lambda i: idxs[i])
    sorted_user_baskets = user_baskets_tensor[sorted_idx]

    user = user.to(device)
    pos = pos.to(device)
    sorted_user_baskets = sorted_user_baskets.to(device)

    return user, pos, sorted_user_baskets

class EvalDataset(Dataset):
    def __init__(self, df, user_baskets):
        self.user_baskets = user_baskets
        self.user_ids = sorted(df['user_id'].unique())
        self.user_gt = df.groupby('user_id')['item_id'].apply(set).to_dict()
    
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        return user_id, self.user_baskets[user_id], self.user_gt[user_id]
    
def eval_collate_fn(batch, device):
    user_ids = torch.tensor([b[0] for b in batch], dtype=torch.long)
    baskets = torch.stack([b[1] for b in batch], dim=0)
    gts = [b[2] for b in batch]
    return user_ids.to(device), baskets.to(device), gts

def load_data(dataset):
    df = pd.read_csv(f'../data/{dataset}/baskets_inter.csv')
    train_df = df[df['tag'] == 0]
    test_df = df[df['tag'] == 1]
    val_df = df[df['tag'] == 2]
    num_users = df['user_id'].max() + 1
    num_items = df['item_id'].max() + 1
    
    image_feat_path = f'../data/{dataset}/image_feat.npy'
    text_feat_path = f'../data/{dataset}/text_feat.npy'
    image_feat = np.load(image_feat_path) if os.path.exists(image_feat_path) else None
    text_feat = np.load(text_feat_path) if os.path.exists(text_feat_path) else None

    user_latest_basket = train_df.groupby('user_id')['basket_ord'].max().to_dict()
    return train_df, test_df, val_df, num_users, num_items, image_feat, text_feat, user_latest_basket

def build_graph(train_df, num_users, num_items, user_latest_basket, config, device):
    rows_train, cols_train, vals_train = [], [], []
    rows_test, cols_test, vals_test = [], [], []

    train_basket_dict = defaultdict(lambda: defaultdict(list))
    test_basket_dict = defaultdict(lambda: defaultdict(list))

    user_basket_cnt = defaultdict(int)

    max_baskets = train_df.groupby(['user_id'])['basket_ord'].transform('max')
    is_max = train_df['basket_ord'] == max_baskets

    for idx, row in train_df.iterrows():
        u, i, b_ord = row['user_id'], row['item_id'], row['basket_ord']
        weight = config['ui_decay'] ** (user_latest_basket[u] - b_ord)

        # test_adj is the full graph
        rows_test.append(u)
        cols_test.append(i)
        vals_test.append(weight)
        test_basket_dict[u][b_ord].append(i)
        if user_basket_cnt[u] < 30:
            user_basket_cnt[u] += 1

        # train_adj uses only the non-latest records
        if not is_max.loc[idx]:
            rows_train.append(u)
            cols_train.append(i)
            vals_train.append(weight)
            train_basket_dict[u][b_ord].append(i)

    # === build user_baskets on CPU first, then move everything to GPU ===
    def build_basket_tensor(basket_dict, max_basket_size=30, max_seq_len=30):
        user_baskets = []
        for u in range(num_users):
            baskets = basket_dict.get(u, {})
            sorted_b = sorted(baskets.items())
            
            if len(sorted_b) > max_seq_len:
                sorted_b = sorted_b[-max_seq_len:]
            
            basket_tensors = [torch.tensor(item_list, dtype=torch.long) for _, item_list in sorted_b]
            
            # cap the number of items in a single basket to max_basket_size
            truncated_tensors = []
            for tensor in basket_tensors:
                if tensor.size(0) > max_basket_size:
                    truncated_tensors.append(tensor[-max_basket_size:])
                else:
                    truncated_tensors.append(tensor)
            
            # pad baskets so that all baskets have the same size
            padded_tensors = [
                F.pad(t, (0, max_basket_size - t.size(0)), value=num_items)
                for t in truncated_tensors
            ]
            
            # if there are fewer than max_seq_len baskets, left-pad with empty baskets
            if len(padded_tensors) < max_seq_len:
                padding_size = max_seq_len - len(padded_tensors)
                padding_tensors = [torch.full((max_basket_size,), num_items, dtype=torch.long) for _ in range(padding_size)]
                padded_tensors = padding_tensors + padded_tensors
                basket_tensor = torch.stack(padded_tensors)  # [max_seq_len, max_basket_size]
            else:
                basket_tensor = torch.stack(padded_tensors)  # [seq_len, max_basket_size]
            
            user_baskets.append(basket_tensor)
        
        return torch.stack(user_baskets)
    
    train_user_baskets = build_basket_tensor(train_basket_dict)
    test_user_baskets = build_basket_tensor(test_basket_dict)

    # === fine-grained ablations (separating repetition / time-decay / adaptive-norm) ===
    # config['ablation'] values:
    #   None / 'full'  -> original behavior (Eq.3: accumulate over repeated baskets after lambda1 decay)
    #   'no_rep'       -> drop the repetition accumulation: keep only the decay weight of the most recent
    #                     occurrence of each (u,i) (keeps recency/time-decay, removes repeat-count stacking);
    #                     equivalent to taking the max weight per (u,i)
    #   'no_decay'     -> drop time-decay: equivalent to setting config['ui_decay']=1 (pure repeat count), no need for this branch
    #   'no_norm'      -> drop adaptive-norm: equivalent to setting ui_norm_w=iu_norm_w=-0.5 (symmetric LightGCN)
    # Note: no_decay / no_norm are pure config options; only no_rep, which alters the graph, needs handling here.
    def _dedup_max(rows, cols, vals):
        best = {}
        for r, c, v in zip(rows, cols, vals):
            key = (r, c)
            if key not in best or v > best[key]:
                best[key] = v
        nr, nc, nv = [], [], []
        for (r, c), v in best.items():
            nr.append(r); nc.append(c); nv.append(v)
        return nr, nc, nv

    if config.get('ablation') == 'no_rep':
        rows_train, cols_train, vals_train = _dedup_max(rows_train, cols_train, vals_train)
        rows_test, cols_test, vals_test = _dedup_max(rows_test, cols_test, vals_test)

    # === build the adjacency graph ===
    def build_ui_adj(rows, cols, vals, w):
        edge_index = torch.LongTensor([rows, cols])
        edge_value = torch.FloatTensor(vals)
        return normalize_ui_adj(edge_index, edge_value, num_users, num_items, w).to(device)

    train_ui_adj = build_ui_adj(rows_train, cols_train, vals_train, config['ui_norm_w'])
    test_ui_adj = build_ui_adj(rows_test, cols_test, vals_test, config['ui_norm_w'])

    train_iu_adj = build_ui_adj(rows_train, cols_train, vals_train, config['iu_norm_w'])
    test_iu_adj = build_ui_adj(rows_test, cols_test, vals_test, config['iu_norm_w'])

    # === item-item co-occurrence (complementarity) block: computed from tag0 training baskets, symmetric degree normalization, no leakage ===
    def build_ii_adj(basket_dict, num_items, min_co, device):
        from itertools import combinations
        co = defaultdict(float)
        for u, baskets in basket_dict.items():
            for b_ord, items in baskets.items():
                uniq = sorted(set(int(x) for x in items))
                for a, b in combinations(uniq, 2):
                    co[(a, b)] += 1.0
        rows, cols, vals = [], [], []
        for (a, b), c in co.items():
            if c < min_co:
                continue
            rows += [a, b]; cols += [b, a]; vals += [c, c]   # symmetric
        if not rows:
            return None
        idx = torch.LongTensor([rows, cols]); val = torch.FloatTensor(vals)
        deg = torch.zeros(num_items); deg.scatter_add_(0, idx[0], val)
        dinv = deg.pow(-0.5).nan_to_num_(0.0)
        nval = dinv[idx[0]] * val * dinv[idx[1]]   # D^-1/2 C D^-1/2 symmetric normalization
        return torch.sparse_coo_tensor(idx, nval, (num_items, num_items), device=device).coalesce()

    use_ii = config.get('item_item', False)
    ii_w = float(config.get('ii_weight', 1.0))
    ii_min = float(config.get('ii_min_co', 3))
    train_ii_adj = build_ii_adj(train_basket_dict, num_items, ii_min, device) if use_ii else None
    test_ii_adj = build_ii_adj(test_basket_dict, num_items, ii_min, device) if use_ii else None
    if use_ii:
        _ne = 0 if train_ii_adj is None else train_ii_adj._nnz()
        print(f"[item-item] enabled w={ii_w} min_co={ii_min} train_ii_nnz={_ne}", flush=True)

    def build_adj(ui_adj, iu_adj, num_users, num_items, ii_adj=None):
        ui_indices = ui_adj.indices()
        ui_values = ui_adj.values()
        row_u = ui_indices[0]
        col_i = ui_indices[1] + num_users

        iu_indices = iu_adj.indices()
        iu_values = iu_adj.values()
        row_i = iu_indices[1] + num_users
        col_u = iu_indices[0]

        rows_cat = [row_u, row_i]; cols_cat = [col_i, col_u]; vals_cat = [ui_values, iu_values]
        if ii_adj is not None:
            ii_idx = ii_adj.indices(); ii_val = ii_adj.values() * ii_w
            rows_cat.append(ii_idx[0] + num_users); cols_cat.append(ii_idx[1] + num_users); vals_cat.append(ii_val)

        all_rows = torch.cat(rows_cat)
        all_cols = torch.cat(cols_cat)
        all_values = torch.cat(vals_cat)

        adj_size = num_users + num_items
        return torch.sparse_coo_tensor(
            torch.stack([all_rows, all_cols]),
            all_values,
            size=(adj_size, adj_size),
            device=ui_adj.device
        ).coalesce()

    train_adj = build_adj(train_ui_adj, train_iu_adj, num_users, num_items, train_ii_adj)
    test_adj = build_adj(test_ui_adj, test_iu_adj, num_users, num_items, test_ii_adj)

    return train_adj, test_adj, train_user_baskets, test_user_baskets, user_basket_cnt