# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import r2_score

# -----------------------------
# tools: CI / RM2
# -----------------------------
def concordance_index(event, pred):
    event = np.asarray(event).ravel()
    pred = np.asarray(pred).ravel()
    n = 0
    n_concordant = 0.0
    n_tied = 0.0
    N = len(event)
    for i in range(N):
        for j in range(i + 1, N):
            if event[i] == event[j]:
                continue
            n += 1
            if (event[i] > event[j] and pred[i] > pred[j]) or (event[i] < event[j] and pred[i] < pred[j]):
                n_concordant += 1
            elif pred[i] == pred[j]:
                n_tied += 1
    if n == 0:
        return 0.5
    return (n_concordant + 0.5 * n_tied) / n

def r_squared_error(y_obs, y_pred):
    y_obs = np.asarray(y_obs, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_obs.size == 0 or y_pred.size == 0:
        return 0.0
    y_obs_mean = np.mean(y_obs)
    y_pred_mean = np.mean(y_pred)
    num = np.sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    num = num * num
    denom = np.sum((y_obs - y_obs_mean) ** 2) * np.sum((y_pred - y_pred_mean) ** 2)
    if denom <= 1e-12:
        return 0.0
    return num / float(denom)

def get_k(y_obs, y_pred):
    y_obs = np.asarray(y_obs, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    denom = np.sum(y_pred * y_pred)
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(y_obs * y_pred) / denom)

def squared_error_zero(y_obs, y_pred):
    y_obs = np.asarray(y_obs, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_obs.size == 0 or y_pred.size == 0:
        return 0.0
    k = get_k(y_obs, y_pred)
    y_obs_mean = np.mean(y_obs)
    upp = np.sum((y_obs - (k * y_pred)) ** 2)
    down = np.sum((y_obs - y_obs_mean) ** 2)
    if down <= 1e-12:
        return 0.0
    return 1.0 - (upp / float(down))

def get_rm2(y_obs, y_pred):
    y_obs = np.asarray(y_obs, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    if y_obs.size == 0 or y_pred.size == 0:
        return 0.0
    r2 = r_squared_error(y_obs, y_pred)
    r02 = squared_error_zero(y_obs, y_pred)
    value = (r2 * r2) - (r02 * r02)
    if value < 0:
        value = 0.0
    return r2 * (1.0 - np.sqrt(np.abs(value)))

# -----------------------------
# utils: mask from lengths
# -----------------------------
def lengths_to_mask(lens, max_len):
    if lens is None:
        return None
    device = lens.device
    idxs = torch.arange(max_len, device=device).unsqueeze(0)
    mask = idxs >= lens.unsqueeze(1)  # True 表示 padding 位置
    return mask

# -----------------------------
# Attention Pooling
# -----------------------------
class AttnPool(nn.Module):
    def __init__(self, dim, n_heads=1, dropout=0.1):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, 1, dim))
        self.proj = nn.Linear(dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: (B, L, D), mask: (B, L) True for pad
        B, L, D = x.shape
        q = self.q.expand(B, 1, D)               # (B,1,D)
        k = self.proj(x)                         # (B,L,D)
        s = (q.float() @ k.float().transpose(1, 2)) / math.sqrt(D)  # (B,1,L)
        if mask is not None:
            s = s.masked_fill(mask.unsqueeze(1), -1e4)             # fp16 安全
        a = torch.softmax(s, dim=-1).to(x.dtype)                    # (B,1,L)
        a = self.dropout(a)
        out = a @ x                                                 # (B,1,D)
        return out.squeeze(1)                                       # (B,D)

# -----------------------------
# GCN
# -----------------------------
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True, use_gcn_norm=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.use_gcn_norm = use_gcn_norm

    def forward(self, x, adj):
        h = self.linear(x)  # (B, N, Fout)
        if not self.use_gcn_norm:
            return F.relu(torch.bmm(adj, h))
        B, N, _ = x.size()
        I = torch.eye(N, device=x.device, dtype=adj.dtype).unsqueeze(0).expand(B, N, N)
        A = adj + I
        deg = A.sum(-1)
        deg_inv_sqrt = torch.pow(deg + 1e-8, -0.5)
        D = torch.diag_embed(deg_inv_sqrt)
        A_hat = torch.bmm(torch.bmm(D, A), D)
        return F.relu(torch.bmm(A_hat, h))

class DenseGCN(nn.Module):
    def __init__(self, in_features, hid_features=256, n_layers=2, dropout=0.1, use_gcn_norm=True):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            in_f = in_features if i == 0 else hid_features
            self.layers.append(GraphConvolution(in_f, hid_features, use_gcn_norm=use_gcn_norm))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
            x = self.dropout(x)
        return x

# -----------------------------
# TextCNN
# -----------------------------
class TextCNN(nn.Module):
    def __init__(self, embed_dim, hid_dim, kernels=(3,5,7), dropout=0.2, use_bn=False):
        super().__init__()
        blocks = []
        for k in kernels:
            layers = [nn.Conv1d(embed_dim, hid_dim, kernel_size=k, padding=k//2, bias=not use_bn)]
            if use_bn:
                layers += [nn.BatchNorm1d(hid_dim)]
            layers += [nn.ReLU()]
            blocks.append(nn.Sequential(*layers))
        self.convs = nn.ModuleList(blocks)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hid_dim * len(kernels), hid_dim)

    def forward(self, x):              # x: (B, L, D)
        x = x.transpose(1, 2)          # -> (B, D, L)
        feats = [conv(x).amax(dim=2)   # (B, hid)
                 for conv in self.convs]
        z = torch.cat(feats, dim=1)    # (B, hid * |K|)
        out = self.proj(self.dropout(z))
        return out                     # (B, hid)

# -----------------------------
# Decoder (cross attention)
# -----------------------------
class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(hid_dim)
        self.sa = nn.MultiheadAttention(hid_dim, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(hid_dim)
        self.ea = nn.MultiheadAttention(hid_dim, n_heads, dropout=dropout, batch_first=True)
        self.ln3 = nn.LayerNorm(hid_dim)
        self.pf = nn.Sequential(nn.Linear(hid_dim, pf_dim), nn.ReLU(), nn.Linear(pf_dim, hid_dim))
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        trg2, _ = self.sa(trg, trg, trg, key_padding_mask=trg_mask)
        trg2 = self.ln1(trg + self.do(trg2))
        trg3, _ = self.ea(trg2, src, src, key_padding_mask=src_mask)
        trg3 = self.ln2(trg2 + self.do(trg3))
        trg4 = self.ln3(trg3 + self.do(self.pf(trg3)))
        return trg4, src

class Decoder(nn.Module):
    def __init__(self, hid_dim, n_layers, n_heads, pf_dim, dropout=0.1, use_attn_mask=True):
        super().__init__()
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(n_layers)])
        self.use_attn_mask = use_attn_mask

    def forward(self, trg, src, trg_lens=None, src_lens=None):
        if self.use_attn_mask:
            trg_kpm = lengths_to_mask(trg_lens, trg.size(1)) if trg_lens is not None else None
            src_kpm = lengths_to_mask(src_lens, src.size(1)) if src_lens is not None else None
        else:
            trg_kpm, src_kpm = None, None
        for layer in self.layers:
            trg, src = layer(trg, src, trg_kpm, src_kpm)
        return trg, src

# -----------------------------
# Tensor Network
# -----------------------------
class TensorNetworkModule(nn.Module):
    def __init__(self, k_feature, hid_dim, k_dim):
        super().__init__()
        self.fc1 = nn.Linear(hid_dim, k_dim)
        self.fc2 = nn.Linear(k_dim, hid_dim)

    def forward(self, e1, e2):
        if e1.dim() == 3:
            e1 = e1.squeeze(1)
        if e2.dim() == 3:
            e2 = e2.squeeze(1)
        out = self.fc1(e1 * e2)
        out = F.relu(out)
        out = self.fc2(out)
        return out

# -----------------------------
# Multi-Sample Dropout Head
# -----------------------------
class MSDropoutHead(nn.Module):
    def __init__(self, in_dim, n_train_samples=4, n_infer_samples=16, p1=0.4, p2=0.3):
        super().__init__()
        self.n_train = n_train_samples
        self.n_infer = n_infer_samples
        self.fc1 = nn.Linear(in_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.out = nn.Linear(128, 1)
        self.do1 = nn.Dropout(p1)
        self.do2 = nn.Dropout(p2)

    def _forward_once(self, x, force_dropout=False):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.do1.p, training=(self.training or force_dropout))
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.do2.p, training=(self.training or force_dropout))
        return self.out(x)

    def forward(self, x):
        if self.training:
            n = max(1, self.n_train)
            outs = [self._forward_once(x, force_dropout=True) for _ in range(n)]
            return torch.mean(torch.stack(outs, dim=0), dim=0).squeeze(-1)
        else:
            n = max(1, self.n_infer)
            outs = [self._forward_once(x, force_dropout=True) for _ in range(n)]
            return torch.mean(torch.stack(outs, dim=0), dim=0).squeeze(-1)

# -----------------------------
# Predictor
# -----------------------------
class Predictor(nn.Module):
    def __init__(self, gcn, cnn, decoder, inter_att, tensor_network, device,
                 n_fingerprint, n_layers, atom_dim, protein_dim, hid_dim,
                 use_attn_mask=True):
        super().__init__()
        self.gcn = gcn
        self.cnn = cnn
        self.decoder = decoder
        self.tensor_network = tensor_network
        self.device = device
        self.use_attn_mask = use_attn_mask

        self.embed_fingerprint = nn.Embedding(n_fingerprint, atom_dim, padding_idx=0)
        self.fc_fpt = nn.Linear(atom_dim, hid_dim)
        self.fc_protein = nn.Linear(protein_dim, hid_dim)

        self.pool_comp1 = AttnPool(hid_dim, dropout=0.1)
        self.pool_prot1 = AttnPool(hid_dim, dropout=0.1)
        self.pool_comp2 = AttnPool(hid_dim, dropout=0.1)

        self.out_head = MSDropoutHead(hid_dim * 6, n_train_samples=4, n_infer_samples=16, p1=0.4, p2=0.3)

    def forward(self, compound1, compound2_idx, adj, protein1, comp1_lens=None, prot1_lens=None):
        # graph branch
        compound1_gcn = self.gcn(compound1, adj)  # (B, N, hid)
        comp1_mask = lengths_to_mask(comp1_lens, compound1_gcn.size(1)) if self.use_attn_mask else None
        comp1_pool_attn = self.pool_comp1(compound1_gcn, comp1_mask)            # (B,hid)

        # fingerprint branch（仅作特征，不拼原始药物2）
        comp2_emb = self.embed_fingerprint(compound2_idx)    # (B, L2, atom_dim)
        comp2_emb_h = self.fc_fpt(comp2_emb)                 # (B, L2, hid)
        comp2_mask = (compound2_idx == 0) if self.use_attn_mask else None
        comp2_pool_attn = self.pool_comp2(comp2_emb_h, comp2_mask)              # (B,hid)

        # protein branch
        protein_h = self.fc_protein(protein1)                # (B, Lp, hid)
        prot1_mask = lengths_to_mask(prot1_lens, protein_h.size(1)) if self.use_attn_mask else None
        prot_cnn = self.cnn(protein_h)                       # (B, hid)

        # decoder
        dec_trg, dec_src = self.decoder(protein_h, compound1_gcn, prot1_lens, comp1_lens)
        prot_pool_attn = self.pool_prot1(dec_trg, prot1_mask)               # (B,hid)
        comp_pool_attn = self.pool_comp1(dec_src, comp1_mask)               # (B,hid)

        # tensor net
        tensor_out = self.tensor_network(comp_pool_attn, prot_pool_attn)     # (B,hid)

        x = torch.cat([
            comp1_pool_attn,
            comp2_pool_attn,
            prot_cnn,
            comp_pool_attn,
            prot_pool_attn,
            tensor_out
        ], dim=-1)
        out = self.out_head(x)  # (B,)
        return out

# -----------------------------
# EMA
# -----------------------------
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

# -----------------------------
# Trainer（AMP + rank  + RDrop + EMA）
# -----------------------------
class Trainer:
    def __init__(self, model, lr=3e-4, weight_decay=2e-4, device="cuda",
                 accum_steps=1, rdrop_coef=0.02, loss_type="mse",
                 use_rank_loss=True, rank_lambda=0.05, rank_pairs=512, use_amp=True,
                 use_ema=True, ema_decay=0.999, total_epochs=160, warmup_epochs=8):
        self.model = model
        self.device = device
        self.accum_steps = accum_steps
        self.rdrop_coef = rdrop_coef
        self.use_rank_loss = use_rank_loss
        self.rank_lambda = rank_lambda
        self.rank_pairs = rank_pairs
        self.use_amp = use_amp
        self.use_ema = use_ema
        self.ema = EMA(model, decay=ema_decay) if use_ema else None
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.cur_epoch = 1

        self.criterion = nn.MSELoss() if loss_type.lower() == "mse" else nn.SmoothL1Loss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = None
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def set_epoch(self, ep: int):
        self.cur_epoch = ep

    def _rank_weight(self):
        # 10~(T*0.6)
        T = self.total_epochs
        s, e = 10, int(T * 0.6)
        if self.cur_epoch <= s:
            return 0.0
        if self.cur_epoch >= e:
            # 0
            rem = T - e
            t = max(0, min(rem, self.cur_epoch - e))
            w = 0.5 * (1 + math.cos(math.pi * t / max(1, rem)))
            return self.rank_lambda * w
        return self.rank_lambda * (self.cur_epoch - s) / max(1, (e - s))

    def _pairwise_rank_loss(self, y, y_hat):
        if (not self.use_rank_loss) or y.size(0) < 2 or self.rank_pairs <= 0:
            return y_hat.new_tensor(0.0)
        B = y.size(0)
        i = torch.randint(0, B, (self.rank_pairs,), device=y.device)
        j = torch.randint(0, B, (self.rank_pairs,), device=y.device)
        neq = (i != j)
        i, j = i[neq], j[neq]
        if i.numel() == 0:
            return y_hat.new_tensor(0.0)
        s = torch.sign(y[i] - y[j])
        non_zero = (s != 0)
        if non_zero.sum() == 0:
            return y_hat.new_tensor(0.0)
        i, j, s = i[non_zero], j[non_zero], s[non_zero]
        diff = (y_hat[i] - y_hat[j]) * s
        return torch.mean(torch.log1p(torch.exp(-diff)))  # logistic

    def train(self, loader, device):
        self.model.train()
        total_loss, n_batches = 0.0, 0
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader):
            comp1, comp2_idx, adj, prot1, prot2, labels, comp1_lens, prot1_lens = batch
            comp1 = comp1.to(device); comp2_idx = comp2_idx.to(device)
            adj = adj.to(device); prot1 = prot1.to(device)
            labels = labels.to(device).float()
            comp1_lens = comp1_lens.to(device); prot1_lens = prot1_lens.to(device)

            with torch.cuda.amp.autocast(enabled=self.use_amp):
                out1 = self.model(comp1, comp2_idx, adj, prot1, comp1_lens, prot1_lens)
                if self.rdrop_coef > 0:
                    out2 = self.model(comp1, comp2_idx, adj, prot1, comp1_lens, prot1_lens)
                    loss_main = 0.5 * (self.criterion(out1, labels) + self.criterion(out2, labels)) \
                                + self.rdrop_coef * F.mse_loss(out1, out2)
                    out_for_rank = out1
                else:
                    loss_main = self.criterion(out1, labels)
                    out_for_rank = out1

                rank_w = self._rank_weight()
                loss_rank = self._pairwise_rank_loss(labels.detach(), out_for_rank) if rank_w > 0 else out1.new_tensor(0.0)
                loss = loss_main + rank_w * loss_rank
                loss = loss / float(self.accum_steps)

            self.scaler.scale(loss).backward()
            if (step + 1) % self.accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.ema is not None:
                    self.ema.update(self.model)

            total_loss += loss.item()
            n_batches += 1

        if self.scheduler is not None:
            self.scheduler.step()
        return total_loss / max(1, n_batches)

# -----------------------------
# Tester（EMA）
# -----------------------------
class Tester:
    def __init__(self, model, device, ema: "EMA" = None):
        self.model = model
        self.device = device
        self.ema = ema

    def test(self, loader, use_ema: bool = True):
        if self.ema is not None and use_ema:
            self.ema.apply_shadow(self.model)
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in loader:
                comp1, comp2_idx, adj, prot1, prot2, labels, comp1_lens, prot1_lens = batch
                comp1 = comp1.to(self.device)
                comp2_idx = comp2_idx.to(self.device)
                adj = adj.to(self.device)
                prot1 = prot1.to(self.device)
                labels = labels.to(self.device).float()
                comp1_lens = comp1_lens.to(self.device)
                prot1_lens = prot1_lens.to(self.device)

                outputs = self.model(comp1, comp2_idx, adj, prot1, comp1_lens, prot1_lens)
                preds.extend(outputs.detach().cpu().numpy().reshape(-1).tolist())
                trues.extend(labels.detach().cpu().numpy().reshape(-1).tolist())

        if self.ema is not None and use_ema:
            self.ema.restore(self.model)

        if len(trues) == 0:
            return float("nan"), float("nan"), float("nan"), float("nan")
        preds = np.asarray(preds, dtype=float).ravel()
        trues = np.asarray(trues, dtype=float).ravel()
        mse = float(np.mean((trues - preds) ** 2))
        ci = float(concordance_index(trues, preds))
        rm2 = float(get_rm2(trues, preds))
        r2 = float(r2_score(trues, preds)) if trues.size > 1 else 0.0
        return mse, ci, rm2, r2


__all__ = [
    "DenseGCN", "GraphConvolution",
    "TextCNN", "Decoder", "TensorNetworkModule",
    "Predictor", "Trainer", "Tester", "EMA"
]
