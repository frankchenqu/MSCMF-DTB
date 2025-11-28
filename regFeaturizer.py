# -*- coding: utf-8 -*-

import os, json, h5py, pickle, warnings
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdchem, rdPartialCharges
import torch
from tqdm import tqdm

# ========= warning & RDKit log =========
warnings.filterwarnings("ignore", category=DeprecationWarning)
RDLogger.DisableLog('rdApp.*')

# ========= TAPE =========
try:
    from tape import TAPETokenizer, ProteinBertModel
    def load_protein_model():
        print("使用 ProteinBertModel (tape)")
        return ProteinBertModel.from_pretrained("bert-base"), TAPETokenizer(vocab="iupac")
except Exception:
    def load_protein_model():
        print("未找到 TAPE，跳过 BERT embedding")
        return None, None

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def enhanced_atom_features(atom, ring_sizes_of_atom, gasteiger_charge):
    symbol = ['C','N','O','F','P','S','Cl','Br','I','other']  # 10
    degree = [0,1,2,3,4,5,6]                                  # 7
    hyb = [
        rdchem.HybridizationType.SP,
        rdchem.HybridizationType.SP2,
        rdchem.HybridizationType.SP3,
        rdchem.HybridizationType.SP3D,
        rdchem.HybridizationType.SP3D2,
        'other'
    ]                                                         # 6

    feats = []
    feats += one_of_k_encoding_unk(atom.GetSymbol(), symbol)  # 10
    feats += one_of_k_encoding_unk(atom.GetDegree(), degree)  # 7
    feats += [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()]  # 2
    feats += one_of_k_encoding_unk(atom.GetHybridization(), hyb)      # 6
    feats += [atom.GetIsAromatic()]                                  # 1  —— 累计 26

    total_h = int(atom.GetTotalNumHs(includeNeighbors=True))
    total_h = max(0, min(total_h, 4))
    feats += [int(total_h == k) for k in range(5)]                    # +5 => 31

    chi = atom.GetChiralTag()
    chi_set = [
        rdchem.ChiralType.CHI_UNSPECIFIED,
        rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        rdchem.ChiralType.CHI_OTHER
    ]
    feats += one_of_k_encoding_unk(chi, chi_set)                      # +4 => 35

    in_ring = int(atom.IsInRing())
    feats += [in_ring]                                                # +1 => 36
    sizes = [0,0,0,0,0,0]  # 3,4,5,6,7,8+
    for s in ring_sizes_of_atom:
        if   s == 3: sizes[0] = 1
        elif s == 4: sizes[1] = 1
        elif s == 5: sizes[2] = 1
        elif s == 6: sizes[3] = 1
        elif s == 7: sizes[4] = 1
        elif s >= 8: sizes[5] = 1
    feats += sizes                                                    # +6 => 42

    chg = float(gasteiger_charge) if gasteiger_charge is not None else 0.0
    mass = atom.GetMass() / 100.0
    feats += [chg, mass]                                               # +2 => 44

    return feats  #  44

# -------------------------
# adj
# -------------------------
def weighted_adjacency(mol):
    N = mol.GetNumAtoms()
    adj = np.zeros((N, N), dtype=np.float32)
    for b in mol.GetBonds():
        i = b.GetBeginAtomIdx()
        j = b.GetEndAtomIdx()
        bt = b.GetBondType()
        if b.GetIsAromatic() or bt == rdchem.BondType.AROMATIC:
            w = 1.5
        elif bt == rdchem.BondType.SINGLE:
            w = 1.0
        elif bt == rdchem.BondType.DOUBLE:
            w = 2.0
        elif bt == rdchem.BondType.TRIPLE:
            w = 3.0
        else:
            w = 1.0
        adj[i, j] = w
        adj[j, i] = w
    for i in range(N):
        adj[i, i] = 1.0
    return adj

# -------------------------
# Finger
# -------------------------
def ecfp_indices(mol, nBits=8192, radius=2, useHs=True, useChirality=True):
    if useHs:
        mol = Chem.AddHs(mol)
    try:
        from rdkit.Chem import rdFingerprintGenerators
        gen = rdFingerprintGenerators.GetMorganGenerator(
            radius=radius, fpSize=nBits, useChirality=useChirality
        )
        fp = gen.GetFingerprint(mol)
        onbits = list(fp.GetOnBits())
    except Exception:
        from rdkit.Chem import AllChem
        bitvect = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius=radius, nBits=nBits, useChirality=useChirality
        )
        onbits = list(bitvect.GetOnBits())

    if len(onbits) == 0:
        return np.array([1], dtype=np.int64)  # 避免空序列；0 保留 padding
    return np.array([b + 1 for b in onbits], dtype=np.int64)

# -------------------------
# Y
# -------------------------
def load_Y_matrix(y_path, dataset):
    with open(y_path, "rb") as f:
        Y = pickle.load(f, encoding="latin1")
    Y = np.array(Y, dtype=np.float32)
    if dataset.lower() == "davis":
        Y = -np.log10(Y / 1e9)
    return Y

# -------------------------
# main
# -------------------------
if __name__ == "__main__":
    DATASET = "Davis"   # or KIBA
    base_dir = f"dataset/{DATASET}"
    lig_path = os.path.join(base_dir, "ligands_can.txt")
    prot_path = os.path.join(base_dir, "proteins.txt")
    y_path = os.path.join(base_dir, "Y")

    with open(lig_path, "r") as f:
        ligands = json.load(f)
    with open(prot_path, "r") as f:
        proteins = json.load(f)
    prot_ids = list(proteins.keys())

    print(f"加载到 {len(ligands)} ligands, {len(prot_ids)} proteins")

    Y = load_Y_matrix(y_path, DATASET)
    print("Y shape:", Y.shape)
    if Y.shape[0] != len(ligands):
        Y = Y.T
    assert Y.shape[0] == len(ligands) and Y.shape[1] == len(prot_ids)

    data = []
    lig_keys = list(ligands.keys())
    for i, lig in enumerate(lig_keys):
        for j, pid in enumerate(prot_ids):
            v = Y[i, j]
            if not np.isnan(v):
                data.append((ligands[lig], proteins[pid], float(v)))
    print(f"总样本数: {len(data)}")

    out_dir = os.path.join(base_dir, "bert-modify-MDL-CPI")
    os.makedirs(out_dir, exist_ok=True)
    h5_path = os.path.join(out_dir, "all_data.h5")
    if os.path.exists(h5_path): os.remove(h5_path)

    f = h5py.File(h5_path, "w")
    dt_float32 = h5py.special_dtype(vlen=np.dtype('float32'))
    dt_int64   = h5py.special_dtype(vlen=np.dtype('int64'))

    d_comp1 = f.create_dataset("compounds1", (len(data),), dtype=dt_float32)
    d_comp1_shape = f.create_dataset("compounds1_shape", (len(data),2), dtype=np.int64)
    d_comp2 = f.create_dataset("compounds2", (len(data),), dtype=dt_int64)
    d_adj = f.create_dataset("adjacencies", (len(data),), dtype=dt_float32)
    d_adj_shape = f.create_dataset("adjacencies_shape", (len(data),2), dtype=np.int64)
    d_prot1 = f.create_dataset("proteins1", (len(data),), dtype=dt_float32)
    d_prot1_shape = f.create_dataset("proteins1_shape", (len(data),2), dtype=np.int64)
    d_prot2 = f.create_dataset("proteins2", (len(data),), dtype=dt_int64)
    d_label = f.create_dataset("interactions", (len(data),), dtype=np.float32)

    prot_model, tokenizer = load_protein_model()
    if prot_model is not None:
        prot_model.eval()

    protein_cache = {}

    # === tqdm ===
    for idx, (smi, seq, label) in enumerate(tqdm(data, desc="处理样本")):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"无法解析 SMILES: {smi}")

            try:
                rdPartialCharges.ComputeGasteigerCharges(mol)
                charges = [float(a.GetProp("_GasteigerCharge")) if a.HasProp("_GasteigerCharge") else 0.0
                           for a in mol.GetAtoms()]
            except Exception:
                charges = [0.0] * mol.GetNumAtoms()

            ring_sizes_of_atom = [[] for _ in range(mol.GetNumAtoms())]
            for ring in mol.GetRingInfo().AtomRings():
                sz = len(ring)
                for ai in ring:
                    ring_sizes_of_atom[ai].append(sz)

            atom_feat = np.array(
                [enhanced_atom_features(a, ring_sizes_of_atom[a.GetIdx()], charges[a.GetIdx()])
                 for a in mol.GetAtoms()],
                dtype=np.float32
            )
            adj = weighted_adjacency(mol).astype(np.float32)
            fp_idx = ecfp_indices(mol, nBits=8192, radius=2, useHs=True, useChirality=True)

            if seq in protein_cache:
                prot_emb = protein_cache[seq]
            else:
                if prot_model is not None and tokenizer is not None:
                    token_ids = np.array(tokenizer.encode(seq), dtype=np.int64)
                    token_ids_tensor = torch.tensor(token_ids[None,:], dtype=torch.long)
                    input_mask = torch.ones_like(token_ids_tensor)
                    with torch.no_grad():
                        output = prot_model(token_ids_tensor, input_mask)
                    prot_emb = output[0].cpu().numpy().reshape(-1, 768).astype(np.float32)
                else:
                    prot_emb = np.zeros((max(1,len(seq)-2), 768), dtype=np.float32)
                protein_cache[seq] = prot_emb

            d_comp1[idx] = atom_feat.flatten()
            d_comp1_shape[idx] = atom_feat.shape
            d_comp2[idx] = fp_idx
            d_adj[idx] = adj.flatten()
            d_adj_shape[idx] = adj.shape
            d_prot1[idx] = prot_emb.flatten()
            d_prot1_shape[idx] = prot_emb.shape
            words = np.array([hash(seq[i:i+3]) % 10000 for i in range(max(1, len(seq)-2))], dtype=np.int64)
            d_prot2[idx] = words
            d_label[idx] = float(label)

        except Exception as e:
            print(f"跳过样本 {idx}: {e}")
            continue

        if (idx+1) % 200 == 0:
            f.flush()

    f.close()
    print(f"HDF5 已保存: {h5_path}")
