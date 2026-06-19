# Data

Experiments use three Amazon review subsets — **Beauty**, **Grocery**, **Sports**
(McAuley et al., SIGIR 2016). Raw data is not committed here; regenerate it with the
notebooks in `../preprocessing/` (run `0_gen_inter_csv` → `1_gen_baskets` → `2_gen_feats`).

Expected layout after preprocessing (one folder per dataset):

```
data/
├── beauty/
│   ├── baskets_inter.csv      # columns: user_id, item_id, basket_ord, tag (0=history, 1=target)
│   ├── image_feat.npy         # 4096-d visual features, indexed by item_id
│   └── text_feat.npy          # 384-d textual features (Sentence-BERT all-MiniLM-L6-v2), indexed by item_id
├── grocery/
└── sports/
```
