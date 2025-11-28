# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
global_file_index = 1
import numpy as np
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score,
    precision_recall_curve, auc, log_loss, accuracy_score, f1_score
)


def process_attention_weights(attention_weights):
    global global_file_index
    attention_weights_head_0 = attention_weights[:, 0, :, :]  # [1, 8，114, 31] -> [1,114,31]

    min_values, _ = torch.min(attention_weights_head_0, dim=2, keepdim=True)  # [1,114,1]
    max_values, _ = torch.max(attention_weights_head_0, dim=2, keepdim=True)  # [1,114,1]

    normalized_attention_weights = (attention_weights_head_0 - min_values) / (
                max_values - min_values + 1e-10)

    aggregated_attention = torch.sum(normalized_attention_weights, dim=2)  # [1, 114]
    # max and min
    min_agg_value, _ = torch.min(aggregated_attention, dim=1, keepdim=True)  # [1, 1]
    max_agg_value, _ = torch.max(aggregated_attention, dim=1, keepdim=True)  # [1, 1]

    aggregated_attention_normalized = (aggregated_attention - min_agg_value) / (
                max_agg_value - min_agg_value + 1e-10)  # [1, 114]

    # print("Aggregated Normalized Attention Weights:", aggregated_attention_normalized)
    output_dir = "weight"
    # output_file_path = os.path.join(output_dir, f"human_normalized_attention_weights_{global_file_index}.txt")
    output_file_path = os.path.join(output_dir, f"new_weights_d2_{global_file_index}.txt")

    # 检查并创建目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    global_file_index = global_file_index + 1

    attention_values = aggregated_attention_normalized.cpu().detach().numpy().flatten()  # 转换为 NumPy 数组并展平

    with open(output_file_path, 'w') as f:
        for value in attention_values:
            f.write(f"{value}\n")

    print(f"Aggregated Normalized Attention Weights saved to {output_file_path}")

# -----------------------------
# Utility
# -----------------------------
def masked_mean(tensor, mask=None, dim=1, keepdim=False):
    if mask is None:
        return tensor.mean(dim=dim, keepdim=keepdim)
    mask = mask.unsqueeze(-1).float()
    masked = tensor * mask
    summed = masked.sum(dim=dim, keepdim=keepdim)
    denom = mask.sum(dim=dim, keepdim=keepdim).clamp(min=1e-8)
    return summed / denom


# -----------------------------
# Graph Convolution with residual
# -----------------------------
class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x, adj):
        # x: (B, N, in_features)
        support = self.linear(x)                  # (B, N, out_features)
        out = torch.bmm(adj, support)             # (B, N, out_features)
        out = F.relu(out)
        out = self.dropout(out)
        # residual if same size
        if self.in_features == self.out_features:
            # project skip if needed to match type
            skip = x
            return out + skip
        return out


class DenseGCN(nn.Module):
    def __init__(self, in_features, hid_features, n_layers=2, dropout=0.0):
        super().__init__()
        layers = []
        for i in range(n_layers):
            in_f = in_features if i == 0 else hid_features
            layers.append(GraphConvolution(in_f, hid_features, dropout))
        self.layers = nn.ModuleList(layers)

    def forward(self, x, adj):
        for layer in self.layers:
            x = layer(x, adj)
        return x  # (B, N, hid_features)


# -----------------------------
# Self-Attention & Decoder (return pooled representations)
# -----------------------------
class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.tensor(hid_dim // n_heads, dtype=torch.float32))

    def forward(self, query, key, value, mask=None):
        bsz = query.size(0)
        Q = self.w_q(query).view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0,2,1,3)
        K = self.w_k(key).view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0,2,1,3)
        V = self.w_v(value).view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0,2,1,3)
        energy = torch.matmul(Q, K.permute(0,1,3,2)) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask[:, None, None, :] == 0, -1e10)

        energy1 = F.softmax(energy, dim=-1)
        if (energy1.shape[2]> energy1.shape[3]):
            process_attention_weights(energy1)


        attn = torch.matmul(F.softmax(energy, dim=-1), V)
        attn = attn.permute(0, 2, 1, 3).contiguous().view(bsz, -1, self.hid_dim)
        return self.fc(self.do(attn))


class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc1 = nn.Conv1d(hid_dim, pf_dim, 1)
        self.fc2 = nn.Conv1d(pf_dim, hid_dim, 1)
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.do(F.relu(self.fc1(x)))
        x = self.fc2(x)
        x = x.permute(0,2,1)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(hid_dim)
        self.sa = SelfAttention(hid_dim, n_heads, dropout)
        self.ln2 = nn.LayerNorm(hid_dim)
        self.ea = SelfAttention(hid_dim, n_heads, dropout)
        self.ln3 = nn.LayerNorm(hid_dim)
        self.pf = PositionwiseFeedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        trg2 = self.ln1(trg + self.do(self.sa(trg, trg, trg, trg_mask)))
        trg3 = self.ln2(trg2 + self.do(self.ea(trg2, src, src, src_mask)))
        trg4 = self.ln3(trg3 + self.do(self.pf(trg3)))

        src2 = self.ln1(src + self.do(self.sa(src, src, src, src_mask)))
        src3 = self.ln2(src2 + self.do(self.ea(src2, trg, trg, trg_mask)))
        src4 = self.ln3(src3 + self.do(self.pf(src3)))

        # return token-level updated sequences
        return trg4, src4


class Decoder(nn.Module):
    def __init__(self, hid_dim, n_layers, n_heads, pf_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(hid_dim, n_heads, pf_dim, dropout) for _ in range(n_layers)]
        )

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        # trg: (B, L, hid), src: (B, N, hid)
        for layer in self.layers:
            trg, src = layer(trg, src, trg_mask, src_mask)
        # pooled outputs (B,1,hid)
        m1 = torch.mean(trg, dim=1, keepdim=True)
        m2 = torch.mean(src, dim=1, keepdim=True)
        return m1, m2


class TextCNN(nn.Module):
    def __init__(self, embed_dim, hid_dim, kernels=[3,5,7], dropout=0.2):
        super().__init__()
        # Conv1d: 输入形状 (B, D, L)，沿 L 卷积；D 当作 in_channels
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels=embed_dim, out_channels=hid_dim, kernel_size=k, padding=k//2),
                nn.ReLU()
            )
            for k in kernels
        ])
        self.do = nn.Dropout(dropout)
        self.fc = nn.Linear(hid_dim * len(kernels), hid_dim)

    def forward(self, protein):                  # protein: (B, L, D)
        x = protein.transpose(1, 2)              # -> (B, D, L)
        feats = [conv(x).max(dim=2).values for conv in self.convs]  # [(B, hid_dim), ...]
        z = torch.cat(feats, dim=1)              # (B, hid_dim * |K|)
        out = self.fc(self.do(z))                # (B, hid_dim)
        return out.unsqueeze(1)                  # (B, 1, hid_dim)


# -----------------------------
# Interaction Model
# -----------------------------
class InteractionModel(nn.Module):
    def __init__(self, hid_dim, n_heads):
        super().__init__()
        self.compound_attention = nn.MultiheadAttention(hid_dim, n_heads)
        self.protein_attention = nn.MultiheadAttention(hid_dim, n_heads)
        self.compound_fc = nn.Linear(hid_dim, hid_dim)
        self.protein_fc = nn.Linear(hid_dim, hid_dim)
        self.activation = nn.ReLU()

    def forward(self, compound_features, protein_features):
        comp = self.activation(compound_features).permute(1,0,2)
        prot = self.activation(protein_features).permute(1,0,2)
        comp_att_out, _ = self.compound_attention(comp, comp, comp)
        prot_att_out, _ = self.protein_attention(prot, prot, prot)
        comp_att_out = comp_att_out.permute(1,0,2)
        prot_att_out = prot_att_out.permute(1,0,2)
        comp_out = self.activation(self.compound_fc(comp_att_out))
        prot_out = self.activation(self.protein_fc(prot_att_out))
        com_att = torch.unsqueeze(torch.mean(comp_out, dim=1), 1)
        pro_att = torch.unsqueeze(torch.mean(prot_out, dim=1), 1)
        return com_att, pro_att


# -----------------------------
# Tensor Network
# -----------------------------
class TensorNetworkModule(nn.Module):
    def __init__(self, k_feature, hid_dim, k_dim):
        super().__init__()
        self.fc1 = nn.Linear(hid_dim, k_dim)
        self.fc2 = nn.Linear(k_dim, hid_dim)

    def forward(self, e1, e2):
        e1 = e1.squeeze(1)
        e2 = e2.squeeze(1)
        out = self.fc1(e1 * e2)
        out = torch.relu(out)
        out = self.fc2(out)
        return out.unsqueeze(1)


# -----------------------------
# Predictor (improved)
# -----------------------------
class Predictor(nn.Module):
    def __init__(self, gcn, cnn, decoder, inter_att, tensor_network,
                 device, n_fingerprint, n_layers,
                 atom_dim=34, protein_dim=768, hid_dim=64):
        super().__init__()
        self.embed_fingerprint = nn.Embedding(n_fingerprint, atom_dim)
        self.gcn = gcn
        self.prot_textcnn = cnn
        self.inter_att = inter_att
        self.tensor_network = tensor_network
        self.decoder = decoder
        self.device = device

        # project protein 768 -> hid (single linear)
        self.fc_protein = nn.Linear(protein_dim, hid_dim)
        # project raw atom features -> hid before gcn
        self.fc_atom = nn.Linear(atom_dim, hid_dim)
        # project fingerprint embedding -> hid
        self.fc_fpt = nn.Linear(atom_dim, hid_dim)

        # classifier: 更适中深度（避免过拟合但有表达力）
        self.out = nn.Sequential(
            nn.Linear(hid_dim * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, compound1, compound2_idx, adj, protein1, protein2=None):
        # compound1: (B,N,atom_dim)
        # protein1: (B,L,768)
        # project protein -> hid
        protein_h = self.fc_protein(protein1)    # (B,L,hid)

        # atom features -> hid -> GCN
        compound1_h = self.fc_atom(compound1)    # (B,N,hid)
        compound1_gcn = self.gcn(compound1_h, adj)  # (B,N,hid)

        # decoder returns pooled representations (B,1,hid)
        protein_pool, compound_pool = self.decoder(protein_h, compound1_gcn)  # (B,1,hid),(B,1,hid)

        # compound2 branch
        compound2_emb = self.embed_fingerprint(compound2_idx)  # (B,N2,atom_dim)
        compound2_emb_h = self.fc_fpt(compound2_emb)           # (B,N2,hid)

        # protein cnn branch (feed protein_h)
        protein_cnn = self.prot_textcnn(protein_h)             # (B,1,hid)

        # pooled features
        comp1_pool = compound1_gcn.mean(dim=1)                 # (B,hid)
        comp2_pool = compound2_emb_h.mean(dim=1)               # (B,hid)
        prot_cnn_p = protein_cnn.squeeze(1)                    # (B,hid)
        comp_pool_p = compound_pool.squeeze(1)                 # (B,hid)
        prot_pool_p = protein_pool.squeeze(1)                  # (B,hid)

        # tensor network over decoder pooled outputs
        tensor_out = self.tensor_network(compound_pool, protein_pool).squeeze(1)  # (B,hid)

        out_fc = torch.cat([
            comp1_pool,comp2_pool,prot_cnn_p, comp_pool_p, prot_pool_p, tensor_out
        ], dim=-1)

        out = self.out(out_fc)
        return out


# -----------------------------
# Trainer & Tester
# -----------------------------
class Trainer(object):
    def __init__(self, model, lr, weight_decay, batch_size, device, accum_steps=1):
        self.model = model
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.batch_size = batch_size
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.accum_steps = accum_steps
        self.scaler = torch.cuda.amp.GradScaler()
        self.autocast = lambda: torch.cuda.amp.autocast()
        self.scheduler = None

    def train(self, dataloader, device):
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        self.optimizer.zero_grad()
        for step, batch in enumerate(dataloader):
            compound1, compound2_idx, adj, protein1, protein2, labels = batch
            compound1 = compound1.to(device)
            compound2_idx = compound2_idx.to(device)
            adj = adj.to(device)
            protein1 = protein1.to(device)
            labels = labels.to(device).long()

            with self.autocast():
                outputs = self.model(compound1, compound2_idx, adj, protein1)
                loss = self.criterion(outputs, labels) / self.accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.accum_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                if self.scheduler is not None:
                    try:
                        self.scheduler.step()
                    except Exception:
                        pass

            total_loss += loss.item() * self.accum_steps
            n_batches += 1

        return total_loss / max(1, n_batches)


class Tester(object):
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.autocast = lambda: torch.cuda.amp.autocast()

    def test(self, dataloader):
        self.model.eval()
        T, Y, S = [], [], []
        with torch.no_grad():
            for batch in dataloader:
                compound1, compound2_idx, adj, protein1, protein2, labels = batch
                compound1 = compound1.to(self.device)
                compound2_idx = compound2_idx.to(self.device)
                adj = adj.to(self.device)
                protein1 = protein1.to(self.device)
                labels = labels.to(self.device).long()

                with self.autocast():
                    outputs = self.model(compound1, compound2_idx, adj, protein1)
                    probs = F.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)

                S.extend(probs[:, 1].cpu().numpy().tolist())
                Y.extend(preds.cpu().numpy().tolist())
                T.extend(labels.cpu().numpy().tolist())

        T = np.array(T); Y = np.array(Y); S = np.array(S)
        if len(T) == 0:
            return 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

        if len(np.unique(T)) == 1:
            roc_auc, pr_auc, log_auc, acc = 0.5, 0.0, log_loss(T, S), accuracy_score(T, Y)
            precision = precision_score(T, Y, zero_division=0)
            recall = recall_score(T, Y, zero_division=0)
            f1 = f1_score(T, Y, zero_division=0)
        else:
            roc_auc = roc_auc_score(T, S)
            precision_curve, recall_curve, _ = precision_recall_curve(T, S)
            pr_auc = auc(recall_curve, precision_curve)
            log_auc = log_loss(T, S)
            acc = accuracy_score(T, Y)
            precision = precision_score(T, Y, zero_division=0)
            recall = recall_score(T, Y, zero_division=0)
            f1 = f1_score(T, Y, zero_division=0)

        return roc_auc, pr_auc, log_auc, acc, precision, recall, f1


# Exports
__all__ = [
    "DenseGCN", "GraphConvolution",
    "TextCNN", "Decoder", "InteractionModel",
    "TensorNetworkModule", "Predictor", "Trainer", "Tester"
]
