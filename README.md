# MT²N: Multimodal Next-Basket Recommendation with Repetition-Aware Time-Decay and Behavior-Popularity Tuning

MT²N is a next-basket recommendation (NBR) model that brings **multimodal item content**
(visual + textual) to the *exploration* component of a basket — the items a user has never
purchased before, which carry little collaborative signal for the ID embedding. It couples content with an
**NBR-specific user–item graph** that uses **repetition-aware time-decayed edge weights** and a
**tunable, direction-specific behavior–popularity normalization**.

## Repository structure

```
MT2N/
├── src/
│   ├── main.py        # CLI entry point
│   ├── model.py       # MT²N model: ID-gated multimodal fusion, modality-specific
│   │                  #   graph-to-sequence encoding, cross-modal contrastive fusion
│   ├── procedure.py   # training / validation / test loop; per-dataset best configs
│   ├── dataset.py     # data loading, basket sequences, NBR-specific graph construction
│   └── utils.py       # metrics (Recall@k, NDCG@k) and history-length group evaluation
├── preprocessing/
│   ├── 0_gen_inter_csv.ipynb     # raw reviews → interaction CSV
│   ├── 1_gen_baskets.ipynb       # interactions → per-user basket sequences
│   ├── 2_gen_feats.ipynb         # visual / textual feature extraction
│   └── 3_augment_text_feat.ipynb # optional textual-feature augmentation
└── data/                         # processed datasets go here (see data/README.md)
```

## Requirements

- Python 3.8+
- PyTorch (CUDA recommended)
- numpy, pandas, scipy

## Data

Experiments use three Amazon review subsets — **Beauty**, **Grocery**, and **Sports**
(McAuley et al., *Image-based recommendations on styles and substitutes*, SIGIR 2016).
Run the `preprocessing/` notebooks in order (0 → 1 → 2) to turn the raw reviews into the
interaction CSV, per-user basket sequences, and the 4096-d visual and 384-d textual item features.
Each processed dataset is expected under `data/<dataset>/` (e.g. `data/beauty/`). We retain
users/items with ≥10 interactions and users with ≥3 baskets, and use the standard
leave-last-basket-out split (the interaction graph is built only from non-final baskets to
prevent leakage; the held-out last baskets are split 50/50 into test and validation by user).

## Training and evaluation

The per-dataset hyperparameters in `BEST_CONFIGS` (in `src/procedure.py`) reproduce the paper's
reported results. From the repository root:

```bash
cd src
python main.py --dataset beauty     # or: grocery | sports
```

Each run trains with early stopping on the validation set and reports test Recall@{10,20} and
NDCG@{10,20}, plus history-length group breakdowns.

### Ablations

```bash
python main.py --dataset grocery --ablation no_rep    # drop repetition aggregation
python main.py --dataset grocery --ablation no_decay  # drop time decay (λ₁ = 1)
python main.py --dataset grocery --ablation no_norm   # symmetric normalization (w₁ = w₂ = −0.5)
```

Finer-grained component switches (drop multimodal content, the ID-gate, the fusion gate, a
single modality, the sequence encoder, or the contrastive loss) are exposed as config flags in
`src/model.py` / `src/procedure.py`.

## Notes

- Use `--seed <n>` to run a specific random seed.
- Optional beyond-accuracy metrics (popularity, repeat/explore recall) are computed when an
  `integrate` helper is importable; otherwise they are silently skipped and core
  training/evaluation is unaffected.
