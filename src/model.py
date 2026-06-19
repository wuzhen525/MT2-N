import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import info_nce
import numpy as np

class Model(nn.Module):
    def __init__(self, num_users, num_items, config, image_feat, text_feat, device):
        super().__init__()
        self.device = device
        self.num_users = num_users
        self.num_items = num_items
        self.embed_dim = config['embed_dim']
        self.n_layers = config['n_layers']
        self.decay_rb = config['decay_rb']
        self.cl_loss = config['cl_loss']
        self.neg_num = config.get('neg_num', 0)  # 0 = positive-only (orig); >0 = BPR negative sampling
        self.no_mm = config.get('no_mm', False)  # True = drop visual+textual (ID-only), for w/o mm ablation / explore-driver test
        self.no_gate = config.get('no_gate', False)  # drop the early ID-gate: use raw projected modality embeddings (no ID * sigmoid)
        self.no_fgate = config.get('no_fgate', False) # drop the fusion gate: sum modalities with equal weight (no learned g_img/g_txt weights)
        self.no_v = config.get('no_v', False)        # drop the visual modality
        self.no_t = config.get('no_t', False)        # drop the textual modality
        self.no_seq = config.get('no_seq', False)    # drop sequence encoding: mean-pool over baskets
        self.attn_scale = config.get('attn_scale', False)  # True = scale basket cross-attn by 1/sqrt(d)
        self.save_emb = config.get('save_emb', False)  # gate the t-SNE .npy dumps (parallel-safe: off by default)
        self.emb_tag = config.get('dataset_name', 'x')  # dataset-tagged dir for trained-emb dump

        self.user_id_embed = nn.Embedding(num_users, self.embed_dim)
        self.item_id_embed = nn.Embedding(num_items, self.embed_dim)
        nn.init.xavier_uniform_(self.user_id_embed.weight)
        nn.init.xavier_uniform_(self.item_id_embed.weight)
        self.register_buffer('zero_user_embed', torch.zeros(num_users, self.embed_dim))

        self.image_feat = nn.Parameter(torch.tensor(image_feat, dtype=torch.float32))
        self.img_proj = nn.Linear(image_feat.shape[1], self.embed_dim)
        self.text_feat = nn.Parameter(torch.tensor(text_feat, dtype=torch.float32))
        self.txt_proj = nn.Linear(text_feat.shape[1], self.embed_dim)

        self.gate_v = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim), nn.Sigmoid())
        self.gate_t = nn.Sequential(nn.Linear(self.embed_dim, self.embed_dim), nn.Sigmoid())

        self.Wq_id = nn.Linear(self.embed_dim, self.embed_dim)
        self.Wk_id = nn.Linear(self.embed_dim, self.embed_dim)
        self.Wq_img = nn.Linear(self.embed_dim, self.embed_dim)
        self.Wk_img = nn.Linear(self.embed_dim, self.embed_dim)
        self.Wq_txt = nn.Linear(self.embed_dim, self.embed_dim)
        self.Wk_txt = nn.Linear(self.embed_dim, self.embed_dim)

        self.g_img = nn.Sequential(nn.Linear(self.embed_dim * 2, 1), nn.Sigmoid())
        self.g_txt = nn.Sequential(nn.Linear(self.embed_dim * 2, 1), nn.Sigmoid())

        # analysis variant: add a learnable intra-basket positional encoding so aggregation becomes order-sensitive (off by default = unordered set)
        self.intra_pos = config.get('intra_pos', False)
        if self.intra_pos:
            self.intra_pos_embed = nn.Embedding(64, self.embed_dim)
            nn.init.zeros_(self.intra_pos_embed.weight)  # zero init: start from an unordered set, learn non-zero only if order helps

    def gcn_propagate(self, all_embed, adj, n_layers):
        all_embeddings = [all_embed]
        for _ in range(n_layers):
            all_embed = torch.sparse.mm(adj, all_embed)
            all_embeddings.append(all_embed)
        return torch.stack(all_embeddings, dim=1).mean(dim=1)
    
    def cross_attention_basket(self, query, item_embed, baskets, Wq, Wk):
        mask = (baskets != self.num_items)  # [batch_size, num_baskets, basket_len]
        all_padded = ~mask.any(dim=-1)  # [batch_size, num_baskets]
        
        items = item_embed[baskets]  # [batch_size, num_baskets, basket_len, embed_dim]
        key_in = items
        if self.intra_pos:  # inject intra-basket position into the key -> make attention weights order-sensitive (control experiment, value magnitude unchanged)
            _L = items.shape[2]
            _pos = torch.arange(_L, device=items.device)
            key_in = items + self.intra_pos_embed(_pos).view(1, 1, _L, -1)

        key = Wk(key_in)  # [batch_size, num_baskets, basket_len, embed_dim]
        value = items     # [batch_size, num_baskets, basket_len, embed_dim]
        q = Wq(query).unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, embed_dim]
        
        attn_scores = (q * key).sum(dim=-1)  # [batch_size, num_baskets, basket_len]
        if self.attn_scale:  # scaled dot-product q·k/sqrt(d) (WISE experiment)
            attn_scores = attn_scores / (self.embed_dim ** 0.5)
        attn_scores = attn_scores.masked_fill(~mask, float('-inf'))
        
        attn_weights = torch.softmax(attn_scores, dim=-1).unsqueeze(-1)  # [batch_size, num_baskets, basket_len, 1]
        
        basket_embeds = (attn_weights * value).sum(dim=2)  # [batch_size, num_baskets, embed_dim]
        basket_embeds[all_padded] = 0.0
        
        return basket_embeds

    def encode_sequence(self, basket_embeds):
        num_baskets = basket_embeds.shape[1]
        if self.no_seq:  # drop sequence encoding: uniform mean pooling (no recency decay weighting)
            return basket_embeds.mean(dim=1)
        t = torch.arange(num_baskets, device=self.device, dtype=torch.float32).view(1, num_baskets, 1)
        weights = self.decay_rb ** (num_baskets - 1 - t)  # [1, num_baskets, 1]
        
        weighted_sum = (basket_embeds * weights).sum(dim=1)  # [batch_size, embed_dim]
        sum_weights = weights.sum()
        
        user_embeds = weighted_sum / sum_weights
        
        return user_embeds

    def forward(self, adj, batch_users, user_baskets, is_training=False):
        # gated fusion
        item_img_embed = self.img_proj(self.image_feat)
        item_txt_embed = self.txt_proj(self.text_feat)
        img_proj_raw = item_img_embed   # pre-gate content representation (used for the Fig1 cross-modal CKA geometry analysis, not in the main loop)
        txt_proj_raw = item_txt_embed
        if not self.no_gate:
            item_img_embed = self.item_id_embed.weight * self.gate_v(item_img_embed)
            item_txt_embed = self.item_id_embed.weight * self.gate_t(item_txt_embed)
        # no_gate: use the raw projected modality embeddings directly (no ID gating)

        # GCN
        id_embed = torch.cat([self.user_id_embed.weight, self.item_id_embed.weight], dim=0)
        id_embed = self.gcn_propagate(id_embed, adj, self.n_layers)
        id_users, id_items = torch.split(id_embed, [self.num_users, self.num_items], dim=0)

        img_embed = torch.cat([self.zero_user_embed, item_img_embed], dim=0)
        img_embed = self.gcn_propagate(img_embed, adj, self.n_layers)
        img_users, img_items = torch.split(img_embed, [self.num_users, self.num_items], dim=0)

        txt_embed = torch.cat([self.zero_user_embed, item_txt_embed], dim=0)
        txt_embed = self.gcn_propagate(txt_embed, adj, self.n_layers)
        txt_users, txt_items = torch.split(txt_embed, [self.num_users, self.num_items], dim=0)

        id_users = id_users[batch_users]
        img_users = img_users[batch_users]
        txt_users = txt_users[batch_users]

        # basket aggregation
        basket_id = self.cross_attention_basket(id_users, F.pad(id_items, (0, 0, 0, 1)), user_baskets, self.Wq_id, self.Wk_id)
        basket_img = self.cross_attention_basket(img_users, F.pad(img_items, (0, 0, 0, 1)), user_baskets, self.Wq_img, self.Wk_img)
        basket_txt = self.cross_attention_basket(txt_users, F.pad(txt_items, (0, 0, 0, 1)), user_baskets, self.Wq_txt, self.Wk_txt)

        # sequence encoding
        user_embed_id = self.encode_sequence(basket_id)
        user_embed_img = self.encode_sequence(basket_img)
        user_embed_txt = self.encode_sequence(basket_txt)

        # Modality-Aware Fusion
        w1u = self.g_img(torch.cat([user_embed_id, user_embed_img], dim=1))
        w2u = self.g_txt(torch.cat([user_embed_id, user_embed_txt], dim=1))
        w1i = self.g_img(torch.cat([id_items, img_items], dim=1))
        w2i = self.g_txt(torch.cat([id_items, txt_items], dim=1))
        if self.no_fgate:  # fusion-gate ablation: sum modalities with equal weight
            w1u = w2u = w1i = w2i = 1.0
        if self.no_mm:  # w/o multimodal: ID-only final representation (paper's w/o mm)
            user_embed = user_embed_id
            item_embed = id_items
        else:
            iv = 0.0 if self.no_v else 1.0   # no_v: drop visual
            it = 0.0 if self.no_t else 1.0   # no_t: drop textual
            user_embed = user_embed_id + iv * w1u * user_embed_img + it * w2u * user_embed_txt
            item_embed = id_items + iv * w1i * img_items + it * w2i * txt_items

        tmp_user_embed = torch.zeros(self.num_users, self.embed_dim, device=user_embed.device)
        tmp_user_embed[batch_users] = user_embed
        user_embed = tmp_user_embed

        if is_training:
            if self.save_emb:  # gated; dataset-tagged dir to avoid parallel races
                import os as _os
                _d = 'dumps/trained_emb/%s' % getattr(self, 'emb_tag', 'x')
                _os.makedirs(_d, exist_ok=True)
                _pre = '' if getattr(self, '_emb_dumped', False) else 'init_'  # first call = before training (init)
                np.save('%s/%sid_items.npy' % (_d, _pre), id_items.cpu().detach().numpy())
                np.save('%s/%simg_items.npy' % (_d, _pre), img_items.cpu().detach().numpy())
                np.save('%s/%stxt_items.npy' % (_d, _pre), txt_items.cpu().detach().numpy())
                np.save('%s/%simg_proj_items.npy' % (_d, _pre), img_proj_raw.cpu().detach().numpy())
                np.save('%s/%stxt_proj_items.npy' % (_d, _pre), txt_proj_raw.cpu().detach().numpy())
                self._emb_dumped = True  # later calls overwrite the final dump (no prefix)
            return user_embed, item_embed, user_embed_id, user_embed_img, user_embed_txt, id_items, img_items, txt_items
        else:
            return user_embed, item_embed
    
    def compute_loss(self, user, pos, batch_user_baskets, adj):
        batch_users = torch.unique(user)
        batch_items = torch.unique(pos)

        user_embed, item_embed, user_embed_id, user_embed_img, user_embed_txt, id_items, img_items, txt_items = self.forward(adj, batch_users, batch_user_baskets, is_training=True)

        user_embed = F.normalize(user_embed, dim=-1)
        item_embed = F.normalize(item_embed, dim=-1)

        # NLL Loss
        pos_scores = (user_embed[user] * item_embed[pos]).sum(dim=-1)
        if self.neg_num > 0:
            # BPR negative sampling variant:
            # sample neg_num negatives per positive and apply pairwise BPR loss.
            neg = torch.randint(0, self.num_items, (user.size(0), self.neg_num), device=user.device)
            neg_scores = (user_embed[user].unsqueeze(1) * item_embed[neg]).sum(dim=-1)  # [N, neg_num]
            nll_loss = -F.logsigmoid(pos_scores.unsqueeze(1) - neg_scores).mean()
        else:
            nll_loss = F.binary_cross_entropy_with_logits(pos_scores, torch.ones_like(pos_scores))

        # Modality alignment loss (InfoNCE)        
        user_modal = (user_embed_img + user_embed_txt) / 2
        user_cl_loss = info_nce(user_embed_id, user_modal) + info_nce(user_embed_img, user_embed_txt)

        item_modal = (img_items + txt_items) / 2
        item_cl_loss = info_nce(id_items[batch_items], item_modal[batch_items]) + info_nce(img_items[batch_items], txt_items[batch_items])

        cl_loss = user_cl_loss + item_cl_loss

        if self.no_mm:  # no modalities -> no cross-modal alignment loss
            return nll_loss
        return nll_loss + self.cl_loss * cl_loss
    
    def predict(self, adj, user_loader):
        all_scores = []
        all_gts = []

        with torch.no_grad():
            for batch_users, batch_user_baskets, batch_gts in user_loader:
                batch_users = batch_users.to(self.device)
                scores = self.predict_single_batch(adj, batch_users, batch_user_baskets)
                all_scores.append(scores.cpu())
                all_gts.extend(batch_gts)

        all_scores = torch.cat(all_scores, dim=0)
        return all_scores, all_gts

    def predict_single_batch(self, adj, batch_users, user_baskets):
        user_embed, item_embed = self.forward(adj, batch_users, user_baskets, is_training=False)
        user_embed = F.normalize(user_embed[batch_users], dim=-1)
        item_embed = F.normalize(item_embed, dim=-1)
        scores = user_embed @ item_embed.T
        return scores