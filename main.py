# -*- coding: utf-8 -*-
"""
"""
import os
import time
import random
import h5py
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model import (
    DenseGCN, TextCNN, Decoder, InteractionModel,
    TensorNetworkModule, Predictor, Trainer, Tester
)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# ========== Dataset ==========
class CPIDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.h5_file = None
        with h5py.File(h5_path, "r") as f:
            self.n = f["interactions"].shape[0]
        print("总样本数:", self.n)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, "r")

        f = self.h5_file
        comp1_flat = torch.tensor(f["compounds1"][idx], dtype=torch.float32)
        comp1_shape = tuple(f["compounds1_shape"][idx].tolist())
        comp1 = comp1_flat.view(*comp1_shape)

        comp2 = torch.tensor(f["compounds2"][idx], dtype=torch.long)

        adj_flat = torch.tensor(f["adjacencies"][idx], dtype=torch.float32)
        adj_shape = tuple(f["adjacencies_shape"][idx].tolist())
        adj = adj_flat.view(*adj_shape)

        prot1_flat = torch.tensor(f["proteins1"][idx], dtype=torch.float32)
        prot1_shape = tuple(f["proteins1_shape"][idx].tolist())
        prot1 = prot1_flat.view(*prot1_shape)

        prot2 = torch.tensor(f["proteins2"][idx], dtype=torch.long)

        label = torch.tensor(f["interactions"][idx], dtype=torch.long)

        return comp1, comp2, adj, prot1, prot2, label


# ========== collate_fn ==========
def collate_fn(batch):
    compounds1, compounds2, adjacencies, proteins1, proteins2, labels = zip(*batch)

    max_c1 = max(c.size(0) for c in compounds1)
    comp1_batch = torch.stack([F.pad(c, (0, 0, 0, max_c1 - c.size(0))) for c in compounds1])

    max_c2 = max(c.size(0) for c in compounds2)
    comp2_batch = torch.stack([F.pad(c, (0, max_c2 - c.size(0))) for c in compounds2]).long()

    max_adj = max(a.size(0) for a in adjacencies)
    adj_batch = torch.stack([F.pad(a, (0, max_adj - a.size(0), 0, max_adj - a.size(0))) for a in adjacencies]).float()

    max_p1 = min(512, max(p.size(0) for p in proteins1))
    prot1_batch = []
    for p in proteins1:
        if p.size(0) > max_p1:
            p = p[:max_p1, :]
        prot1_batch.append(F.pad(p, (0, 0, 0, max_p1 - p.size(0))))
    prot1_batch = torch.stack(prot1_batch).float()

    max_p2 = max(p.size(0) for p in proteins2)
    prot2_batch = torch.stack([F.pad(p, (0, max_p2 - p.size(0))) for p in proteins2]).long()

    labels = torch.stack(labels).long()
    return comp1_batch, comp2_batch, adj_batch, prot1_batch, prot2_batch, labels


# ========== Main ==========
def main():
    SEED = 1
    random.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    DATASET = "cele"
    h5_path = f"dataset/{DATASET}/bert-modify-MDL-CPI/all_data.h5"
    assert os.path.exists(h5_path), f"HDF5 not found: {h5_path}"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dataset = CPIDataset(h5_path)
    n = len(dataset)
    n_train = int(0.8 * n)
    n_dev = int(0.1 * n)
    n_test = n - n_train - n_dev
    print(f"train/dev/test = {n_train}/{n_dev}/{n_test}")

    batch_size = 64
    lr = 5e-4
    weight_decay = 1e-3
    n_epochs = 60
    accum_steps = 4

    train_ds, dev_ds, test_ds = random_split(dataset, [n_train, n_dev, n_test])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate_fn,
                              num_workers=4, pin_memory=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn,
                             num_workers=2, pin_memory=True)

    # model hyperparams
    protein_dim = 768
    atom_dim = 34
    hid_dim = 64

    n_layers = 3
    n_heads = 8
    pf_dim = 256
    dropout = 0.2
    k_feature, k_dim = 16, 16
    n_fingerprint = 10000

    # build modules: now GCN expects atom_dim -> hid_dim (we project before GCN in Predictor)
    gcn = DenseGCN(hid_dim, hid_dim, n_layers=2, dropout=0.1)
    # cnn = TextCNN(hid_dim, hid_dim, kernels=[3,5,7], dropout=dropout)
    cnn = TextCNN(embed_dim=hid_dim, hid_dim=hid_dim, kernels=[3,5,7], wt_levels=3, dropout_rate=dropout)
    decoder = Decoder(hid_dim=hid_dim, n_layers=2 if n_layers >= 2 else 1, n_heads=n_heads, pf_dim=pf_dim, dropout=dropout)
    inter_att = InteractionModel(hid_dim, n_heads)
    tensor_network = TensorNetworkModule(k_feature, hid_dim, k_dim)

    model = Predictor(
        gcn=gcn, cnn=cnn, decoder=decoder, inter_att=inter_att, tensor_network=tensor_network,
        device=device, n_fingerprint=n_fingerprint, n_layers=n_layers,
        atom_dim=atom_dim, protein_dim=protein_dim, hid_dim=hid_dim
    ).to(device)

    trainer = Trainer(model, lr=lr, weight_decay=weight_decay, batch_size=batch_size, device=device, accum_steps=accum_steps)
    tester = Tester(model, device)
    scheduler = ReduceLROnPlateau(trainer.optimizer, mode="max", factor=0.5, patience=3, verbose=True)

    os.makedirs("celeoutput", exist_ok=True)
    file_AUCs = os.path.join("celeoutput", "metrics.txt")
    with open(file_AUCs, "w") as f:
        f.write("epoch\telapsed_total(s)\ttrain_loss\t"
                "dev_auc_roc\tdev_auc_pr\tlog_loss_dev\tacc_dev\t"
                "test_auc_roc\ttest_auc_pr\tlog_loss_test\tacc_test\t"
                "prec_test\trec_test\tf1_test\n")

    best_auc, no_improve = 0.0, 0
    total_time = 0.0

    for ep in range(1, n_epochs + 1):
        start_time = time.time()

        loss_train = trainer.train(train_loader, device)
        auc_dev_roc, auc_dev_pr, log_loss_dev, acc_dev, prec_dev, rec_dev, f1_dev = tester.test(dev_loader)
        auc_test_roc, auc_test_pr, log_loss_test, acc_test, prec_test, rec_test, f1_test = tester.test(test_loader)

        # scheduler step based on dev
        scheduler.step(auc_dev_roc)

        elapsed = time.time() - start_time
        total_time += elapsed

        # save best by dev
        if auc_dev_roc > best_auc:
            best_auc, no_improve = auc_dev_roc, 0
            os.makedirs("celeoutput/model", exist_ok=True)
            torch.save(model.state_dict(), os.path.join("celeoutput/model", "best.pt"))
        else:
            no_improve += 1

        metrics = [
            ep, round(total_time, 2), round(loss_train, 4),
            round(auc_dev_roc, 4), round(auc_dev_pr, 4), round(log_loss_dev, 4), round(acc_dev, 4),
            round(auc_test_roc, 4), round(auc_test_pr, 4), round(log_loss_test, 4), round(acc_test, 4),
            round(prec_test, 4), round(rec_test, 4), round(f1_test, 4)
        ]
        with open(file_AUCs, "a") as f:
            f.write("\t".join(map(str, metrics)) + "\n")

        print(f"Epoch {ep:02d} | time={elapsed:.2f}s (累积 {total_time:.2f}s) | "
              f"train_loss={loss_train:.4f} | dev_auc={auc_dev_roc:.4f} | test_auc={auc_test_roc:.4f}")

        if no_improve >= 8:
            print("早停")
            break

    # final evaluation: load best model and evaluate on test
    best_path = os.path.join("celeoutput/model", "best.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
        model.to(device)
        final = tester.test(test_loader)
        print("Final test metrics (best dev model):", final)

    print("训练结束，最优 dev AUC:", best_auc)


if __name__ == "__main__":
    main()
