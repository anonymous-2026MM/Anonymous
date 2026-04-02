import torch
import torch.nn.utils.rnn as rnn_utils
import torch.nn as nn
import torch.nn.functional as F
import json
import os
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import random
import DiffModel as Diff
from DiffModel import DiffCDR
import itertools

def infoNCE_loss(anchor, positive, negatives, neg_mask, sample_type_mask, temperature=0.5):
    if not sample_type_mask.any():
        return torch.tensor(0.0, device=anchor.device)
    
    anchor = F.normalize(anchor, p=2, dim=-1)
    positive = F.normalize(positive, p=2, dim=-1)
    negatives = F.normalize(negatives, p=2, dim=-1)
    
    pos_sim = (anchor * positive).sum(dim=-1, keepdim=True) / temperature
    neg_sims = torch.bmm(anchor.unsqueeze(1), negatives.transpose(1, 2)).squeeze(1) / temperature
    neg_sims = neg_sims.masked_fill(~neg_mask, -1e9)
    logits = torch.cat([pos_sim, neg_sims], dim=1)
    labels = torch.zeros(sample_type_mask.sum().item(), dtype=torch.long, device=anchor.device)
    
    return F.cross_entropy(logits[sample_type_mask], labels)

def orthogonal_loss(feat1, feat2, slack=0.2):
    feat1 = F.normalize(feat1, p=2, dim=-1)
    feat2 = F.normalize(feat2, p=2, dim=-1) 
    inner_product = (feat1 * feat2).sum(dim=-1)
    return torch.max(inner_product  ** 2 - slack, torch.zeros_like(inner_product)).mean()

# ==================== VAE ====================

class VAEInv(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        hidden_dim = max(input_dim // 2, latent_dim * 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, latent_dim * 2)
        )
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = h[:, :self.latent_dim], h[:, self.latent_dim:]
        return self.reparameterize(mu, logvar), mu, logvar


class VAEMeta(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        hidden_dim = max(input_dim // 2, latent_dim * 2)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),nn.GELU(),
            nn.LayerNorm(hidden_dim),nn.Linear(hidden_dim, latent_dim * 2)
        )
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = h[:, :self.latent_dim], h[:, self.latent_dim:]
        return self.reparameterize(mu, logvar), mu, logvar


class FusionDecoder(nn.Module):
    def __init__(self, latent_dim, input_dim):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.GELU(),
            nn.LayerNorm(latent_dim * 2),
            nn.Linear(latent_dim * 2, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, input_dim // 2),
            nn.GELU(),
            nn.LayerNorm(input_dim // 2),
            nn.Linear(input_dim // 2, input_dim)
        )

    def forward(self, g_inv, g_var):
        return self.decoder(self.fusion(torch.cat([g_inv, g_var], -1)))

# ==================== Phase 1: Disentangle Model ====================

class DecoupleModel(nn.Module):
    def __init__(self, input_dim=128, latent_dim=64):
        super().__init__()
        self.vae_inv = VAEInv(input_dim, latent_dim)
        self.vae_var_src = VAEMeta(input_dim, latent_dim)
        self.vae_var_tgt = VAEMeta(input_dim, latent_dim)
        self.decoder = FusionDecoder(latent_dim, input_dim)
        self.inv_cache = {}
        self.var_cache = {}

    def clear_cache(self):
        self.inv_cache = {}
        self.var_cache = {}
    
    def batch_encode_asins(self, asins, global_feat, asin2idx, asin2domain, device, training=True):
        valid_asins = []
        indices = []
        domains = []
        for asin in asins:
            if asin in asin2idx and asin in asin2domain:
                valid_asins.append(asin)
                indices.append(asin2idx[asin])
                domains.append(asin2domain[asin])
        batch_feats = global_feat[indices]  # shape: [num_valid, input_dim]
        if training:
            inv_vecs, inv_mus, inv_logvars = self.vae_inv(batch_feats)
        else:
            with torch.no_grad():
                inv_vecs, inv_mus, inv_logvars = self.vae_inv(batch_feats)
        var_vecs = torch.zeros_like(inv_vecs)
        src_mask = torch.tensor([d == 'src' for d in domains], device=device, dtype=torch.bool)
        tgt_mask = torch.tensor([d == 'tgt' for d in domains], device=device, dtype=torch.bool)
        if src_mask.any():
            src_feats = batch_feats[src_mask]
            if training:
                src_var_vecs, _, _ = self.vae_var_src(src_feats)
            else:
                with torch.no_grad():
                    src_var_vecs, _, _ = self.vae_var_src(src_feats)
            var_vecs[src_mask] = src_var_vecs
        if tgt_mask.any():
            tgt_feats = batch_feats[tgt_mask]
            if training:
                tgt_var_vecs, _, _ = self.vae_var_tgt(tgt_feats)
            else:
                with torch.no_grad():
                    tgt_var_vecs, _, _ = self.vae_var_tgt(tgt_feats)
            var_vecs[tgt_mask] = tgt_var_vecs
        for i, asin in enumerate(valid_asins):
            if training:
                self.inv_cache[asin] = inv_vecs[i]
                self.var_cache[asin] = var_vecs[i]
            else:
                self.inv_cache[asin] = inv_vecs[i].detach()
                self.var_cache[asin] = var_vecs[i].detach()

    def forward(self, asin_list, src_image_feat, tgt_image_feat):
        inv_vectors = torch.stack([self.inv_cache[asin] for asin in asin_list])
        var_vectors = torch.stack([self.var_cache[asin] for asin in asin_list])
        return inv_vectors, var_vectors

    def decouple_loss(self, image_samples, global_feat, asin2idx, asin2domain, device):
        uids, sample_types, pos1, pos2, negs = image_samples
        batch_size = len(uids)
        all_asins = set(pos1 + pos2)
        for neg_list in negs:
            all_asins.update(neg_list)
        all_asins = list(all_asins)
        
        self.batch_encode_asins(all_asins, global_feat, asin2idx, asin2domain, device, training=True)
        asin_to_idx = {asin: i for i, asin in enumerate(all_asins)}
        inv_pos1 = torch.stack([self.inv_cache[asin] for asin in pos1]).to(device)
        inv_pos2 = torch.stack([self.inv_cache[asin] for asin in pos2]).to(device)
        var_pos1 = torch.stack([self.var_cache[asin] for asin in pos1]).to(device)
        var_pos2 = torch.stack([self.var_cache[asin] for asin in pos2]).to(device)
        max_neg_num = max(len(neg_list) for neg_list in negs) if negs else 0
        if max_neg_num == 0:
            return torch.tensor(0.0, device=device, requires_grad=True), \
                   torch.tensor(0.0, device=device), \
                   torch.tensor(0.0, device=device), \
                   torch.tensor(0.0, device=device)
        
        neg_indices = torch.zeros(batch_size, max_neg_num, dtype=torch.long, device=device)
        neg_mask = torch.zeros(batch_size, max_neg_num, dtype=torch.bool, device=device)
        
        for i, neg_list in enumerate(negs):
            valid_negs = [asin for asin in neg_list]
            if valid_negs:
                idx = torch.tensor([asin_to_idx[asin] for asin in valid_negs[:max_neg_num]], device=device)
                neg_indices[i, :len(idx)] = idx
                neg_mask[i, :len(idx)] = True
        inv_all = torch.stack([self.inv_cache[asin] for asin in all_asins]).to(device)
        var_all = torch.stack([self.var_cache[asin] for asin in all_asins]).to(device)
        
        inv_negs = inv_all[neg_indices]  # [batch_size, max_neg_num, dim]
        var_negs = var_all[neg_indices]
        
        inv_mask = torch.tensor([t == 'inv' for t in sample_types], device=device)
        var_src_mask = torch.tensor([t == 'var_src' for t in sample_types], device=device)
        var_tgt_mask = torch.tensor([t == 'var_tgt' for t in sample_types], device=device)
        
        info_loss = torch.tensor(0.0, device=device)
        valid_count = 0

        inv_weight=1.5
        
        if inv_mask.any():
            info_loss += inv_weight * infoNCE_loss(
                inv_pos1, inv_pos2, inv_negs, neg_mask, inv_mask, temperature=0.2
            )
            valid_count += 1*inv_weight
        
        if var_src_mask.any():
            info_loss += infoNCE_loss(
                var_pos1, var_pos2, var_negs, neg_mask, var_src_mask, temperature=0.2
            )
            valid_count += 1
        
        if var_tgt_mask.any():
            info_loss += infoNCE_loss(
                var_pos1, var_pos2, var_negs, neg_mask, var_tgt_mask, temperature=0.2
            )
            valid_count += 1
        
        info_loss = info_loss / valid_count if valid_count > 0 else torch.tensor(0.0, device=device)
        
        ortho_loss = orthogonal_loss(inv_pos1, var_pos1)
        
        pos1_indices = torch.tensor([asin2idx[p] for p in pos1 if p in asin2idx], device=device)
        pos2_indices = torch.tensor([asin2idx[p] for p in pos2 if p in asin2idx], device=device)
        feat_pos1 = global_feat[pos1_indices]
        feat_pos2 = global_feat[pos2_indices]
        
        recon_1 = self.decoder(inv_pos1[:len(feat_pos1)], var_pos1[:len(feat_pos1)])
        recon_2 = self.decoder(inv_pos2[:len(feat_pos2)], var_pos2[:len(feat_pos2)])
        recon_loss = F.mse_loss(recon_1, feat_pos1) + F.mse_loss(recon_2, feat_pos2)
        
        total_loss = 0.3 * recon_loss + 0.7 * info_loss
        return total_loss, info_loss, ortho_loss, recon_loss

    def save(self, path):
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path, device, input_dim=128, latent_dim=64):
        model = cls(input_dim=input_dim, latent_dim=latent_dim)
        model.load_state_dict(torch.load(path, map_location=device))
        model.to(device).eval()
        print(f"loading disentangle model: {path}")
        return model

class DecDiffCDR(nn.Module):
    def __init__(self, num_steps, diff_dim, input_dim, c_scale, 
                 diff_sample_steps, diff_task_lambda, diff_mask_rate,
                 decouple_model, global_image_feat, asin2idx, asin2domain,
                 emb_dim=64, device='cuda', image_dim=None,
                 pre_item_vec_src=None,       # Base_Space [num_src_items, emb_dim]
                 asin2src_idx=None,
                 keep_trace=True, aggregate_method="attention",task_lambda=0.3,
                 use_al_mlp=True, use_visual_guidance=True, use_collaborative_guidance = True,
                 diff_speed = 0.5,diff_lambda = 0.1):          # ASIN -> src_idx
        super().__init__()
        self.use_visual_guidance = use_visual_guidance
        self.use_collaborative_guidance = use_collaborative_guidance
        self.diff_model = DiffCDR(
            num_steps, diff_dim, input_dim, c_scale,
            diff_sample_steps, diff_task_lambda, diff_mask_rate,
            keep_trace, aggregate_method,use_al_mlp=use_al_mlp,
            diff_speed = diff_speed
        ).to(device)
        
        self.decouple_model = decouple_model
        self.global_image_feat = global_image_feat
        self.asin2idx = asin2idx
        self.task_lambda=task_lambda
        self.diff_lambda = diff_lambda
        self.asin2domain = asin2domain
        self.emb_dim = emb_dim
        self.image_dim = image_dim if image_dim else emb_dim

        self.pre_item_vec_src = pre_item_vec_src
        self.asin2src_idx = asin2src_idx

        self.test_linear = nn.Sequential(
            nn.Linear(emb_dim, emb_dim*4),
            nn.ReLU(),
            nn.Linear(emb_dim*4, emb_dim)
        )

    def encode_src_seq(self, src_seq_asin):
        batch_inv = []
        device = next(self.parameters()).device
        
        for seq in src_seq_asin:
            inv_vectors = []
            for asin in seq:
                if asin != 'MISSING' and asin in self.decouple_model.inv_cache:
                    inv_vectors.append(self.decouple_model.inv_cache[asin].to(device))
                else:
                    inv_vectors.append(torch.zeros(self.emb_dim, device=device))
            batch_inv.append(torch.stack(inv_vectors, dim=0))
        
        padded_batch = rnn_utils.pad_sequence(batch_inv, batch_first=True, padding_value=0.0)

        padded_batch = torch.nn.functional.normalize(padded_batch, p=2, dim=-1)        
        return self.diff_model.behavior_enc(padded_batch)

    def encode_src_seq2(self, src_seq_asin):
        batch_size = len(src_seq_asin)
        device = next(self.parameters()).device
        seq_len = max(len(seq) for seq in src_seq_asin)
        feat_seq = torch.zeros(batch_size, seq_len, self.emb_dim, device=device)
        
        for i, seq in enumerate(src_seq_asin):
            for j, asin in enumerate(seq[:seq_len]):
                if asin != 'MISSING' and asin in self.asin2src_idx:
                    src_idx = self.asin2src_idx[asin]
                    if src_idx < self.pre_item_vec_src.size(0):
                        feat_seq[i, j] = self.pre_item_vec_src[src_idx]
        
        feat_seq = torch.nn.functional.normalize(feat_seq, p=2, dim=-1)
        return self.diff_model.behavior_enc2(feat_seq)

    def compute_loss(self, user_src, user_tgt, src_seq_asin, item_text, y,):
        device = user_src.device
        if not self.use_visual_guidance:
            cond_emb = torch.zeros_like(user_src)
        else:
            cond_emb = self.encode_src_seq(src_seq_asin)      # inv
        if not self.use_collaborative_guidance:
            cond_emb2 = None
        else:
            cond_emb2 = self.encode_src_seq2(src_seq_asin)    # Base_Space
        diff_loss = Diff.diffusion_loss_fn(
            self.diff_model, user_src, user_tgt, 
            cond_emb, None, None, device, False, cond_emb2
        )
        final_emb = Diff.p_sample_loop(
            self.diff_model, cond_emb, user_src, device, cond_emb2
        )

        recon_loss = F.smooth_l1_loss(final_emb, user_tgt)
        sim_loss = 1 - F.cosine_similarity(final_emb, user_tgt, dim=1).mean()
        
        final_emb_norm = F.normalize(final_emb, p=2, dim=1)
        item_text_norm = F.normalize(item_text, p=2, dim=1)
        y_pred = (final_emb_norm * item_text_norm).sum(1) * self.diff_model.scale + self.diff_model.bias
        
        task_loss = (y_pred - y.squeeze().float()).square().mean()
        total = self.diff_lambda* diff_loss + self.task_lambda* recon_loss + (1-self.task_lambda)* task_loss
        
        return {
            'total': total,
            'recon': recon_loss.item() if recon_loss is not None else 0.0,
            'diff': diff_loss.item() if diff_loss is not None else 0.0,
            'task': task_loss.item() if task_loss is not None else 0.0,
        }

    def predict(self, user_src, user_tgt, src_seq_asin, item_text, y):
        device = user_src.device
        
        if not self.use_visual_guidance:
            cond_emb = torch.zeros_like(user_src)
        else:
            cond_emb = self.encode_src_seq(src_seq_asin)
        if not self.use_collaborative_guidance:
            cond_emb2 = None
        else:
            cond_emb2 = self.encode_src_seq2(src_seq_asin)
        noise_ref = torch.zeros_like(user_src)
        final_emb = Diff.p_sample_loop(
            self.diff_model, cond_emb, noise_ref, device, cond_emb2,
        )
        
        final_emb_norm = F.normalize(final_emb, p=2, dim=1)
        item_text_norm = F.normalize(item_text, p=2, dim=1)
        score = (final_emb_norm * item_text_norm).sum(1) * self.diff_model.scale + self.diff_model.bias
        
        return score

class DecoupleUtils:
    @staticmethod
    def load_feat_dict(root, domain, feat_type):
        import glob
        pattern = os.path.join(root, 'raw', f'{domain}_{feat_type}_*.jsonl')
        files = glob.glob(pattern)        
        feat_dict = {}
        with open(files[0], 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                for asin, vec in data.items():
                    feat_dict[asin] = torch.tensor(vec, dtype=torch.float32)
        
        feat_dim = len(next(iter(feat_dict.values()))) if feat_dict else 0
        print(f"loading {domain} {feat_type} feature : num {len(feat_dict)}，dim {feat_dim}")
        return feat_dict, feat_dim

    @staticmethod
    def build_data_loader(meta_path, input_root, src_feat_dict, tgt_feat_dict, batch_size, 
                        task_id=None, ratio=None, feat_type=None, root='./', shuffle=True, use_cache=True,
                        per_user_pos=10, per_pos_neg=6):
        cache_dir = os.path.join(root, 'cache')
        
        def collate_fn(batch):
            uids = torch.tensor([item[0] for item in batch], dtype=torch.long)
            sample_types = [item[1] for item in batch]
            pos1 = [item[2] for item in batch]
            pos2 = [item[3] for item in batch]
            negs = [item[4] for item in batch]
            return uids, sample_types, pos1, pos2, negs
        
        if use_cache and task_id and feat_type:
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, f"{task_id}_{ratio}_{feat_type}_sample.pth")
            if os.path.exists(cache_path):
                print(f"loading cache sample: {cache_path}")
                batches = torch.load(cache_path)
                return DataLoader(batches, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
        meta_data = pd.read_csv(meta_path, header=None, names=['uid', 'iid', 'y', 'src_pos_seq', 'tgt_pos_seq'])
        
        def parse_seq(s):
            return [int(i) for i in s.strip('[]').split(', ') if i.strip().isdigit()] if isinstance(s, str) and s.strip() else []

        with open(os.path.join(input_root, 'id_map.json')) as f:
            id_map = json.load(f)
        idx2asin_src = {v: k for k, v in id_map['iid_dict_src'].items()}
        idx2asin_tgt = {v: k for k, v in id_map['iid_dict_tgt'].items()}
        src_interacted, tgt_interacted = set(), set()
        user_interact_items = {}
        
        for _, row in tqdm(meta_data.iterrows(), desc="building interaction sets"):
            uid = row['uid']
            src_seq = [idx2asin_src[i] for i in parse_seq(row['src_pos_seq']) if i in idx2asin_src]
            tgt_seq = [idx2asin_tgt[i] for i in parse_seq(row['tgt_pos_seq']) if i in idx2asin_tgt]
            
            if src_seq and tgt_seq:
                user_interact_items[uid] = {'src': src_seq, 'tgt': tgt_seq}
                src_interacted.update(src_seq)
                tgt_interacted.update(tgt_seq)

        src_asins = set(src_feat_dict.keys())
        tgt_asins = set(tgt_feat_dict.keys())
        
        global_hard_neg_src = list(src_asins - src_interacted)
        global_hard_neg_tgt = list(tgt_asins - tgt_interacted)
        global_hard_neg_all = list((src_asins.union(tgt_asins)) - src_interacted - tgt_interacted)

        print(f"co-user: {len(user_interact_items)} | src_neg_pool: {len(global_hard_neg_src)}\ntgt_neg_pool: {len(global_hard_neg_tgt)} | all_neg_pool: {len(global_hard_neg_all)}")
        batches = []
        inv_count,src_var_count,tgt_var_count=0,0,0
        for uid, data in tqdm(user_interact_items.items(), desc="generating samples"):
            src_list, tgt_list = data['src'], data['tgt']
            if src_list and tgt_list:
                inv_pairs = random.sample([(s, t) for s in src_list for t in tgt_list], 
                                        min(per_user_pos*2, len(src_list)*len(tgt_list)))
                for s, t in inv_pairs:
                    if random.random()<0.5:
                        batches.append([uid, 'inv', s, t, random.sample(global_hard_neg_all, per_pos_neg)])
                    else:
                        batches.append([uid, 'inv', t, s, random.sample(global_hard_neg_all, per_pos_neg)])
                    inv_count+=1
            if len(src_list) >= 2:
                src_pairs = [(src_list[i], src_list[j]) for i in range(len(src_list)) for j in range(i+1, len(src_list))]
                for pos1, pos2 in random.sample(src_pairs, min(per_user_pos//2, len(src_pairs))):
                    batches.append([uid, 'var_src', pos1, pos2, random.sample(global_hard_neg_src, per_pos_neg)])
                    src_var_count+=1
            if len(tgt_list) >= 2:
                tgt_pairs = [(tgt_list[i], tgt_list[j]) for i in range(len(tgt_list)) for j in range(i+1, len(tgt_list))]
                for pos1, pos2 in random.sample(tgt_pairs, min(per_user_pos//2, len(tgt_pairs))):
                    batches.append([uid, 'var_tgt', pos1, pos2, random.sample(global_hard_neg_tgt, per_pos_neg)])
                    tgt_var_count+=1

        print(f"Sample Total: {len(batches)}, inv: {inv_count}, var_src: {src_var_count}, var_tgt: {tgt_var_count}")

        if use_cache and task_id and feat_type:
            torch.save(batches, cache_path)
            print(f"sample cache saved : {cache_path}")
        return DataLoader(batches, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    
    @staticmethod
    def load_base_vectors(path, device):
        state = torch.load(path, map_location=device)
        return {
            'user_src': state['pre_user_vec_src'],
            'user_tgt': state['pre_user_vec_tgt'],
            'item_src': state['pre_item_vec_src'],
            'item_tgt': state['pre_item_vec_tgt']
        }