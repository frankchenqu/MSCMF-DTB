# -*- coding: utf-8 -*-
import os
import random
import h5py
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from functools import partial

from regModel import (
    DenseGCN, TextCNN, Decoder,
    TensorNetworkModule, Predictor, Trainer, Tester
)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("medium")
except Exception:
    pass

# -------- HDF5 per-worker  --------
_H5 = None
def _worker_init(h5_path, worker_id):
    global _H5
    _H5 = h5py.File(h5_path, "r")

# ================== Dataset ==================
class CPIDataset(Dataset):
    def __init__(self, h5_path, mean_y=None, std_y=None):
        self.h5_path = h5_path
        self._fh = None
        with h5py.File(h5_path, "r") as f:
            self.n = f["interactions"].shape[0]
        self.mean_y, self.std_y = mean_y, std_y
        print("总样本数:", self.n)

    def __len__(self):
        return self.n

    def _file(self):
        global _H5
        if _H5 is not None:
            return _H5
        if self._fh is None:
            self._fh = h5py.File(self.h5_path, "r")
        return self._fh

    def __getitem__(self, idx):
        f = self._file()
        comp1 = torch.tensor(f["compounds1"][idx], dtype=torch.float32).view(*f["compounds1_shape"][idx])
        comp2 = torch.tensor(f["compounds2"][idx], dtype=torch.long)
        adj   = torch.tensor(f["adjacencies"][idx], dtype=torch.float32).view(*f["adjacencies_shape"][idx])
        prot1 = torch.tensor(f["proteins1"][idx], dtype=torch.float32).view(*f["proteins1_shape"][idx])
        prot2 = torch.tensor(f["proteins2"][idx], dtype=torch.long)  # 占位
        label = torch.tensor(f["interactions"][idx], dtype=torch.float32)
        if self.mean_y is not None and self.std_y is not None:
            label = (label - self.mean_y) / self.std_y
        return comp1, comp2, adj, prot1, prot2, label

# ================== collate_fn（512） ==================
def collate_fn(batch):
    compounds1, compounds2, adjacencies, proteins1, proteins2, labels = zip(*batch)

    max_c1 = max(c.size(0) for c in compounds1)
    comp1_batch = torch.stack([F.pad(c, (0, 0, 0, max_c1 - c.size(0))) for c in compounds1])
    comp1_lens = torch.tensor([c.size(0) for c in compounds1], dtype=torch.long)

    max_c2 = max(c.size(0) for c in compounds2)
    comp2_batch = torch.stack([F.pad(c, (0, max_c2 - c.size(0))) for c in compounds2]).long()  # pad=0

    max_adj = max(a.size(0) for a in adjacencies)
    adj_batch = torch.stack([
        F.pad(a, (0, max_adj - a.size(0), 0, max_adj - a.size(0))) for a in adjacencies
    ]).float()

    max_p1_cap = 512
    max_p1 = min(max_p1_cap, max(p.size(0) for p in proteins1))
    prot1_batch, prot1_lens = [], []
    for p in proteins1:
        L = min(p.size(0), max_p1_cap)
        prot1_lens.append(L)
        if p.size(0) > max_p1_cap:
            stride = max(1, p.size(0) // max_p1_cap)
            p = p[::stride, :][:max_p1_cap, :]
        prot1_batch.append(F.pad(p, (0, 0, 0, max_p1 - p.size(0))))
    prot1_batch = torch.stack(prot1_batch).float()
    prot1_lens = torch.tensor(prot1_lens, dtype=torch.long)

    max_p2 = max(p.size(0) for p in proteins2)
    prot2_batch = torch.stack([F.pad(p, (0, max_p2 - p.size(0))) for p in proteins2]).long()

    labels = torch.stack(labels).float()
    return comp1_batch, comp2_batch, adj_batch, prot1_batch, prot2_batch, labels, comp1_lens, prot1_lens

# ================== split loader ==================
def load_split_indices(train_idx_path, test_idx_path):
    train_ids = eval(open(train_idx_path).read().strip())
    test_ids = eval(open(test_idx_path).read().strip())
    if isinstance(train_ids[0], list):
        train_ids = [i for sub in train_ids for i in sub]
    if isinstance(test_ids[0], list):
        test_ids = [i for sub in test_ids for i in sub]
    return np.array(train_ids, dtype=int), np.array(test_ids, dtype=int)

# ================== Main ==================
def main():
    SEED = 42
    random.seed(SEED); torch.manual_seed(SEED); np.random.seed(SEED)

    DATASET = "Davis"
    h5_path = f"dataset/{DATASET}/bert-modify-MDL-CPI/all_data.h5"
    assert os.path.exists(h5_path), f"HDF5 not found: {h5_path}"

    train_ids, test_ids = load_split_indices(
        f"dataset/{DATASET}/folds/train_fold_setting1.txt",
        f"dataset/{DATASET}/folds/test_fold_setting1.txt"
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    with h5py.File(h5_path, "r") as f:
        labels_all = f["interactions"][:]
    labels_train = labels_all[train_ids]
    mean_y, std_y = float(np.mean(labels_train)), float(np.std(labels_train) + 1e-8)
    print(f"标签均值: {mean_y:.4f}, Std: {std_y:.4f}")

    dataset = CPIDataset(h5_path, mean_y=mean_y, std_y=std_y)
    full_train_ds, test_ds = Subset(dataset, train_ids), Subset(dataset, test_ids)
    n_train = int(0.8 * len(full_train_ds))
    n_dev = len(full_train_ds) - n_train
    train_ds, dev_ds = random_split(full_train_ds, [n_train, n_dev])
    print(f"train/dev/test = {len(train_ds)}/{len(dev_ds)}/{len(test_ds)}")


    batch_size   = 64
    lr           = 3e-4
    weight_decay = 2e-4
    n_epochs     = 160
    warmup_ep    = 8
    accum_steps  = 1
    rdrop_coef   = 0.02
    use_amp      = True

    use_gcn_norm  = True
    use_attn_mask = True

    # ---------------- DataLoader ----------------
    worker_init = partial(_worker_init, h5_path)
    common_loader_kw = dict(num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=4,
                            worker_init_fn=worker_init, collate_fn=collate_fn)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **common_loader_kw)
    dev_loader   = DataLoader(dev_ds,   batch_size=batch_size, shuffle=False, **common_loader_kw)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **common_loader_kw)


    protein_dim = 768
    atom_dim    = 44
    hid_dim     = 256
    n_layers, n_heads, pf_dim, dropout = 3, 8, 512, 0.20
    k_feature, k_dim, n_fingerprint = 16, 32, 10000

    gcn = DenseGCN(atom_dim, hid_features=hid_dim, n_layers=2, dropout=0.10, use_gcn_norm=use_gcn_norm)
    cnn = TextCNN(embed_dim=hid_dim, hid_dim=hid_dim)  # 你要换回 WaveletCNN 也只替这里
    decoder = Decoder(hid_dim=hid_dim, n_layers=n_layers, n_heads=n_heads,
                      pf_dim=pf_dim, dropout=dropout, use_attn_mask=use_attn_mask)
    tensor_network = TensorNetworkModule(k_feature, hid_dim, k_dim)

    model = Predictor(
        gcn=gcn, cnn=cnn, decoder=decoder, inter_att=None,
        tensor_network=tensor_network, device=device,
        n_fingerprint=n_fingerprint, n_layers=n_layers,
        atom_dim=atom_dim, protein_dim=protein_dim, hid_dim=hid_dim,
        use_attn_mask=use_attn_mask
    ).to(device)

    trainer = Trainer(
        model, lr=lr, weight_decay=weight_decay, device=device,
        accum_steps=accum_steps, rdrop_coef=rdrop_coef, loss_type="mse",
        use_rank_loss=True, rank_lambda=0.05, rank_pairs=256, use_amp=use_amp,
        use_ema=True, ema_decay=0.999, total_epochs=n_epochs, warmup_epochs=warmup_ep
    )
    tester = Tester(model, device, ema=trainer.ema)


    cosine = CosineAnnealingLR(trainer.optimizer, T_max=(n_epochs - warmup_ep), eta_min=1e-6)
    def lr_lambda(ep):
        if ep < warmup_ep:
            return (ep + 1) / warmup_ep
        return 1.0
    warmup = LambdaLR(trainer.optimizer, lr_lambda=lr_lambda)


    trainer.scheduler = warmup

    # ---------------- train log ----------------
    os.makedirs("output", exist_ok=True)
    file_metrics = os.path.join("output", "metrics_regression.txt")
    with open(file_metrics, "w") as f:
        f.write("epoch\ttrain_loss\tdev_mse\tdev_ci\tdev_rm2\tdev_r2\t"
                "test_mse\ttest_ci\ttest_rm2\ttest_r2\n")

    best_dev_mse, no_improve, patience = float("inf"), 0, 25

    for ep in range(1, n_epochs + 1):
        trainer.set_epoch(ep)
        loss_train = trainer.train(train_loader, device)

        # scheduler
        if ep == warmup_ep:
            trainer.scheduler = cosine
        elif ep > warmup_ep:
            trainer.scheduler.step()

        def eval_and_denorm(loader):
            mse_norm, ci, rm2, r2 = tester.test(loader, use_ema=True)  # EMA + MC Dropout
            return mse_norm * (std_y ** 2), ci, rm2, r2

        mse_dev,  ci_dev,  rm2_dev,  r2_dev  = eval_and_denorm(dev_loader)
        mse_test, ci_test, rm2_test, r2_test = eval_and_denorm(test_loader)

        if mse_dev < best_dev_mse - 1e-6:
            best_dev_mse, no_improve = mse_dev, 0
            torch.save(model.state_dict(), os.path.join("output", f"best_ep{ep}.pt"))
            print(f"[INFO] New best dev MSE: {best_dev_mse:.6f}")
        else:
            no_improve += 1
            print(f"[INFO] no improvement count: {no_improve}/{patience}")

        with open(file_metrics, "a") as f:
            f.write("\t".join(map(str, [
                ep, round(loss_train, 6),
                round(mse_dev, 6), round(ci_dev, 4), round(rm2_dev, 4), round(r2_dev, 4),
                round(mse_test, 6), round(ci_test, 4), round(rm2_test, 4), round(r2_test, 4)
            ])) + "\n")

        print(f"Epoch {ep:03d} | train_loss={loss_train:.5f} "
              f"| dev_mse={mse_dev:.4f} ci={ci_dev:.4f} rm2={rm2_dev:.4f} r2={r2_dev:.4f} "
              f"| test_mse={mse_test:.4f} ci={ci_test:.4f} rm2={rm2_test:.4f} r2={r2_test:.4f}")

        if no_improve >= patience:
            print("早停 triggered."); break

    print("训练结束。最佳 dev MSE:", best_dev_mse)

if __name__ == "__main__":
    main()
