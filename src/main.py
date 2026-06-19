import argparse
from procedure import grid_search

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='beauty | grocery | sports')
    parser.add_argument('--seed', type=int, default=None, help='override seed (multi-seed runs)')
    parser.add_argument('--ablation', type=str, default=None,
                        help='no_rep | no_decay | no_norm (None = full model)')
    parser.add_argument('--neg_num', type=int, default=None,
                        help='BPR negative samples per positive (0/None = orig positive-only loss)')
    parser.add_argument('--attn_scale', action='store_true',
                        help='scale basket cross-attention by 1/sqrt(d)')
    args = parser.parse_args()

    grid_search(args.dataset, seed=args.seed, ablation=args.ablation, neg_num=args.neg_num,
                attn_scale=args.attn_scale)
