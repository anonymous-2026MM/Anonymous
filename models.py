import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import DiffModel as Diff
torch.manual_seed(1)
class LookupEmbedding(torch.nn.Module):
    def __init__(self, uid_all, iid_all, emb_dim):
        super().__init__()
        self.uid_embedding = torch.nn.Embedding(uid_all, emb_dim)
        self.iid_embedding = torch.nn.Embedding(iid_all + 1, emb_dim)

    def forward(self, x):
        uid_emb = self.uid_embedding(x[:, 0].unsqueeze(1))
        iid_emb = self.iid_embedding(x[:, 1].unsqueeze(1))
        emb = torch.cat([uid_emb, iid_emb], dim=1)
        return emb

class FeatrueAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, input_dim)
        )

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon

    def train_model(self, data, device, epochs=30, lr=1e-3):
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        for epoch in range(epochs):
            self.train()
            _, recon = self(data)
            loss = criterion(recon, data)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        self.eval()
        with torch.no_grad():
            compressed_features, _ = self(data)
        return compressed_features
class MetaNet(torch.nn.Module):
    def __init__(self, emb_dim, meta_dim):
        super().__init__()
        self.event_K = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(emb_dim, 1, False))
        self.event_softmax = torch.nn.Softmax(dim=1)
        self.decoder = torch.nn.Sequential(torch.nn.Linear(emb_dim, meta_dim), torch.nn.ReLU(),
                                           torch.nn.Linear(meta_dim, emb_dim * emb_dim))

    def forward(self, emb_fea, seq_index):
        mask = (seq_index == 0).float()
        event_K = self.event_K(emb_fea)
        t = event_K - torch.unsqueeze(mask, 2) * 1e8
        att = self.event_softmax(t)
        his_fea = torch.sum(att * emb_fea, 1)
        output = self.decoder(his_fea)
        return output.squeeze(1)
class MFBasedModel(torch.nn.Module):
    def __init__(self, uid_all, iid_all, emb_dim, meta_dim_0):
        super().__init__()
        self.emb_dim = emb_dim
        self.src_model = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.tgt_model = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.aug_model = LookupEmbedding(uid_all, iid_all, emb_dim)
        self.iid_all=iid_all
        self.register_buffer('pre_user_vec_tgt', None)
        self.register_buffer('pre_item_vec_tgt', None)
        self.register_buffer('pre_user_vec_src', None)
        self.register_buffer('pre_item_vec_src', None)

        self.mapping = torch.nn.Linear(emb_dim, emb_dim, False)
        self.meta_net = MetaNet(emb_dim, meta_dim_0)
        self.behavior_enc = Diff.BehaviorSeq(emb_dim)

    def forward(self, x, stage, device, diff_model=None, is_task=False):
        if stage == 'train_src':
            emb = self.src_model.forward(x)
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return x
        elif stage in ['train_tgt', 'test_tgt']:
            emb = self.tgt_model.forward(x)
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return x
        elif stage in ['train_aug', 'test_aug']:
            emb = self.aug_model.forward(x)
            x = torch.sum(emb[:, 0, :] * emb[:, 1, :], dim=1)
            return x
