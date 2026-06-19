import torch
from itertools import product
import copy
from dataset import load_data, build_graph, TrainDataset, EvalDataset, collate_fn, eval_collate_fn
from utils import set_seed, evaluate, evaluate_group
from model import Model
from torch.utils.data import DataLoader
import time
import os
from datetime import datetime

DUMP_TOPK = os.environ.get('MT2N_DUMP_TOPK', '0') == '1'  # gate central-rescore topk dump
SAVE_EMB = os.environ.get('MT2N_SAVE_EMB', '0') == '1'  # gate trained modality-embedding dump (Fig1)
from functools import partial
import torch.nn.functional as F

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# === Per-dataset best configs (from the commented grids below; verified to
#     reproduce paper Table 2 exactly on Beauty). Selected by dataset name so
#     experiments run via CLI flags without editing this file. ===
_COMMON = dict(seed=42, patience=5, batch_size=2048, eval_batch_size=4096,
               top_k=[10, 20], embed_dim=64, decay_rb=0.8, cl_loss=5e-3,
               lr=1e-2, weight_decay=1e-4)
BEST_CONFIGS = {
    'beauty':  {**_COMMON, 'embed_dim': 64,  'epochs': 13, 'n_layers': 1, 'ui_norm_w': -0.5,  'iu_norm_w': -0.5, 'ui_decay': 0.55},
    'grocery': {**_COMMON, 'embed_dim': 128, 'epochs': 50, 'n_layers': 2, 'ui_norm_w': 0,     'iu_norm_w': -1,   'ui_decay': 0.05},
    'sports':  {**_COMMON, 'embed_dim': 128, 'epochs': 50, 'n_layers': 3, 'ui_norm_w': -0.25, 'iu_norm_w': -1,   'ui_decay': 0.15},
}

# Validation search grid. The BEST_CONFIGS above are the per-dataset winners
# selected on the validation set; reported numbers are the mean over five seeds
# (run via --seed, seed 42 is the default/search seed).
param_grid = {
    'seed': [42],
    'epochs': [50],
    'patience': [5],
    'batch_size': [2048],
    'eval_batch_size': [4096],
    'top_k': [[10, 20]],
    'embed_dim': [64, 128, 256],
    'n_layers': [1, 2, 3, 4],
    'ui_norm_w': [0, -0.25, -0.5, -0.75, -1],
    'iu_norm_w': [0, -0.25, -0.5, -0.75, -1],
    'ui_decay': [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1],
    'decay_rb': [0.8],
    'cl_loss': [5e-4, 1e-3, 5e-3, 1e-2],
    'lr': [1e-3, 5e-3, 1e-2],
    'weight_decay': [1e-4]
}

log_file = None

def setup_logger(dataset_name):
    global log_file
    log_dir = os.path.join("..", "log")
    os.makedirs(log_dir, exist_ok=True)
    time_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"{dataset_name}-{time_str}.txt")

def print_and_log(msg):
    print(msg)
    if log_file:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(msg + '\n')

def train(config, train_df, test_df, val_df, num_users, num_items, image_feat, text_feat, user_latest_basket):
    set_seed(config['seed'])

    train_adj, test_adj, train_user_baskets, test_user_baskets, user_basket_cnt = build_graph(
        train_df,
        num_users,
        num_items,
        user_latest_basket,
        config,
        device=DEVICE
    )

    train_dataset = TrainDataset(train_df, train_user_baskets, num_items)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=partial(collate_fn, device=DEVICE))

    val_dataset = EvalDataset(val_df, test_user_baskets)
    val_loader = DataLoader(val_dataset, batch_size=config['eval_batch_size'], shuffle=False, collate_fn=partial(eval_collate_fn, device=DEVICE))

    test_dataset = EvalDataset(test_df, test_user_baskets)
    test_loader = DataLoader(test_dataset, batch_size=config['eval_batch_size'], shuffle=False, collate_fn=partial(eval_collate_fn, device=DEVICE))

    model = Model(num_users, num_items, config, image_feat, text_feat, DEVICE).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])

    print_and_log("Trainable parameters:")
    trainable_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
            print_and_log(f"  {name}: {list(param.shape)}")
    print_and_log(f"Trainable parameters: {trainable_params:,}")

    best_recall = 0
    patience = 0
    best_model_state = None

    for epoch in range(config['epochs']):
        start_time = time.time()

        model.train()
        total_loss = 0
        
        for user, pos, batch_user_baskets in train_loader:
            loss = model.compute_loss(user, pos, batch_user_baskets, train_adj)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print_and_log(f"[TRAINEPOCH] {epoch} {time.time()-start_time:.3f}")

        model.eval()
        with torch.no_grad():
            scores, gts = model.predict(test_adj, val_loader)
            eval_results = evaluate(scores, gts, config['top_k'])

            recall_str = ", ".join([f"Recall@{k}: {eval_results['recall'][k]:.4f}" for k in config['top_k']])
            ndcg_str = ", ".join([f"NDCG@{k}: {eval_results['ndcg'][k]:.4f}" for k in config['top_k']])

            epoch_time = time.time() - start_time
            print_and_log(f"Epoch {epoch:03d} | Time: {epoch_time:.2f}s | Loss: {avg_loss:.4f} | Val {recall_str}, {ndcg_str}")

            current_score = eval_results['recall'][10] + eval_results['recall'][20]
            if current_score > best_recall:
                best_recall = current_score
                patience = 0
                best_model_state = copy.deepcopy(model.state_dict())

                # user_embed_path = '../data/grocery/user_id_embed_id.pt'
                # item_embed_path = '../data/grocery/item_id_embed_id.pt'
                # user_e, item_e = model(test_adj, torch.unique(user), batch_user_baskets)
                # user_e = F.normalize(user_e)
                # item_e = F.normalize(item_e)
                # torch.save(user_e.detach().cpu(), user_embed_path)
                # torch.save(item_e.detach().cpu(), item_embed_path)
                # print_and_log(f"Saved user and item embeddings.")
            else:
                patience += 1
                if patience >= config['patience']:
                    print_and_log("Early stopping.")
                    break

    model.load_state_dict(best_model_state)
    model.eval()
    with torch.no_grad():
        scores, gts = model.predict(test_adj, test_loader)
        test_results = evaluate(scores, gts, config['top_k'])
        group_avg_results = evaluate_group(scores, gts, config['top_k'], user_basket_cnt)

        # dump top-40 for the unified central rescore (rescore_nbr_topk.py), same format as baselines
        if config.get('dump_topk', False):
            import numpy as _np
            _sc = scores.detach().cpu().numpy() if hasattr(scores, 'detach') else _np.asarray(scores)
            _topk = _np.argsort(-_sc, axis=1)[:, :40].astype(_np.int64)
            _uids = _np.asarray(list(test_loader.dataset.user_ids), dtype=_np.int64)
            _tgts = _np.array([_np.asarray(sorted(g), dtype=_np.int64) for g in gts], dtype=object)
            _sd = config.get('seed', 42); _abl = config.get('ablation')
            _tag = ('_' + _abl) if _abl else ''
            _tag += '' if _sd == 42 else ('_s%d' % _sd)
            import os as _os; _os.makedirs('dumps/nbr_topk', exist_ok=True)
            _out = 'dumps/nbr_topk/MT2N_%s%s.npz' % (config['dataset_name'], _tag)
            _np.savez(_out, user_ids=_uids, topk=_topk, targets=_tgts)
            print_and_log('[dump_topk] wrote %s users=%d topk=%s' % (_out, len(_uids), _topk.shape))

        recall_str = ", ".join([f"Recall@{k}: {test_results['recall'][k]:.4f}" for k in config['top_k']])
        ndcg_str = ", ".join([f"NDCG@{k}: {test_results['ndcg'][k]:.4f}" for k in config['top_k']])
        print_and_log(f"Test {recall_str}, {ndcg_str}")

        print_and_log("[Test Group-wise Results]")
        for group_name, result in group_avg_results.items():
            recall_str = ", ".join([f"Recall@{k}: {result['recall'][k]:.4f}" for k in config['top_k']])
            ndcg_str = ", ".join([f"NDCG@{k}: {result['ndcg'][k]:.4f}" for k in config['top_k']])
            print_and_log(f"  Group {group_name}: {recall_str}, {ndcg_str}")

        # Beyond-accuracy / popularity / repeat-explore metrics
        if config.get('ext_metrics', True):
            try:
                import sys as _sys, os as _os
                _sys.path.append(_os.path.join(_os.path.dirname(__file__), '..', 'eval_ext'))
                from integrate import compute_ext_metrics
                ext = compute_ext_metrics(scores, gts, test_loader.dataset.user_ids,
                                          train_df, num_items, top_k=config['top_k'])
                for k in config['top_k']:
                    line = ", ".join(f"{m}={v:.4f}" for m, v in ext[k].items())
                    print_and_log(f"[Beyond-accuracy @{k}] {line}")
                cold = ", ".join(f"{l}:R={d['recall']:.4f}(n={d['n_gt']})" for l, d in ext['cold_items'].items())
                print_and_log(f"[Cold-item Recall@{max(config['top_k'])}] {cold}")
            except Exception as e:
                print_and_log(f"[Beyond-accuracy] skipped: {e}")

    # also return the best VALIDATION score (sum of val recall@10+@20) so grid_search
    # can select the config on validation, NOT test (avoids test-set leakage).
    return test_results, best_recall

def grid_search(dataset_name, seed=None, ablation=None, grid=None, neg_num=None, attn_scale=False):
    """Run MT2N on a dataset.
    - Default: use the verified per-dataset best config (single run).
    - seed: override seed (for multi-seed significance runs).
    - ablation: 'no_rep' | 'no_decay' | 'no_norm' | None (see dataset.py).
        'no_decay' -> forces ui_decay=1 ; 'no_norm' -> forces ui_norm_w=iu_norm_w=-0.5.
    - grid: optional dict to run a full product grid instead (overrides best config).
    """
    setup_logger(dataset_name)
    train_df, test_df, val_df, num_users, num_items, image_feat, text_feat, user_latest_basket = load_data(dataset_name)

    if grid is not None:
        param_combinations = list(product(*grid.values()))
        param_keys = list(grid.keys())
        configs = [dict(zip(param_keys, v)) for v in param_combinations]
    else:
        base = dict(BEST_CONFIGS[dataset_name])
        if seed is not None:
            base['seed'] = seed
        if neg_num is not None:
            base['neg_num'] = neg_num
        if attn_scale:
            base['attn_scale'] = True
        if ablation:
            base['ablation'] = ablation
            if ablation == 'no_decay':
                base['ui_decay'] = 1.0
            elif ablation == 'no_norm':
                base['ui_norm_w'] = -0.5; base['iu_norm_w'] = -0.5
            elif ablation == 'no_mm':
                base['no_mm'] = True
            elif ablation == 'no_gate':
                base['no_gate'] = True
            elif ablation == 'no_v':
                base['no_v'] = True
            elif ablation == 'no_t':
                base['no_t'] = True
            elif ablation == 'no_seq':
                base['no_seq'] = True
            elif ablation == 'no_cl':
                base['cl_loss'] = 0.0
        configs = [base]

    best_config, best_recall, best_results = None, 0, {}
    for idx, config in enumerate(configs):
        config['dataset_name'] = dataset_name
        config['dump_topk'] = DUMP_TOPK
        config['save_emb'] = SAVE_EMB
        config['item_item'] = os.environ.get('MT2N_ITEM_ITEM', '0') == '1'   # optional item-item co-occurrence (complementarity) graph
        config['ii_weight'] = float(os.environ.get('MT2N_II_W', '1.0'))
        config['ii_min_co'] = float(os.environ.get('MT2N_II_MIN', '3'))
        print_and_log(f"\nRun {idx + 1}/{len(configs)} | dataset={dataset_name} seed={config['seed']} "
                      f"ablation={config.get('ablation')} | config: {config}")
        test_results, val_score = train(config, train_df, test_df, val_df, num_users, num_items, image_feat, text_feat, user_latest_basket)
        print_and_log('[VALSCORE] dim=%s seed=%s val=%.4f test_R20=%.4f' % (config.get('embed_dim'), config.get('seed'), val_score, test_results['recall'][20]))
        # select config by VALIDATION score (tag2), report its TEST (tag1) — no leakage
        if val_score > best_recall:
            best_recall = val_score
            best_results = test_results
            best_config = config

    print_and_log(f"\nBest config: {best_config}")
    recall_str = ", ".join([f"Recall@{k}: {best_results['recall'][k]:.4f}" for k in best_config['top_k']])
    ndcg_str = ", ".join([f"NDCG@{k}: {best_results['ndcg'][k]:.4f}" for k in best_config['top_k']])
    print_and_log(f"Best Test {recall_str}, {ndcg_str}")