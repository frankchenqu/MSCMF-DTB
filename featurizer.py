# -*- coding: utf-8 -*-

import os, gc, h5py
import numpy as np
from collections import defaultdict
from rdkit import Chem

# ========= TAPE =========
try:
    from tape import TAPETokenizer, ProteinBertModel
    def load_protein_model():
        print("使用 ProteinBertModel")
        return ProteinBertModel.from_pretrained("bert-base"), TAPETokenizer(vocab="iupac")
except ImportError:
    from tape import TAPETokenizer, ProteinModel
    def load_protein_model():
        print("使用 ProteinModel.from_pretrained('bert-base')")
        return ProteinModel.from_pretrained("bert-base"), TAPETokenizer(vocab="iupac")

import torch

num_atom_feat = 34
ngram = 3

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}".format(x, allowable_set))
    return [x == s for s in allowable_set]

def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def atom_features(atom, explicit_H=False, use_chirality=True):
    symbol = ['C','N','O','F','P','S','Cl','Br','I','other']
    degree = [0,1,2,3,4,5,6]
    hybridizationType = [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        'other'
    ]
    results = one_of_k_encoding_unk(atom.GetSymbol(), symbol) + \
              one_of_k_encoding(atom.GetDegree(), degree) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), hybridizationType) + [atom.GetIsAromatic()]
    if not explicit_H:
        results += one_of_k_encoding_unk(atom.GetTotalNumHs(), [0,1,2,3,4])
    if use_chirality:
        try:
            results += one_of_k_encoding_unk(atom.GetProp('_CIPCode'), ['R','S']) + \
                       [atom.HasProp('_ChiralityPossible')]
        except:
            results += [False, False] + [atom.HasProp('_ChiralityPossible')]
    return results

def mol_features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    atom_feat = np.zeros((mol.GetNumAtoms(), num_atom_feat))
    for atom in mol.GetAtoms():
        atom_feat[atom.GetIdx(), :] = atom_features(atom)
    adj = Chem.GetAdjacencyMatrix(mol) + np.eye(mol.GetNumAtoms())
    return atom_feat, adj

# ========== main ==========
if __name__ == "__main__":
    DATASET = "drugbank"
    dir_input = f"dataset/{DATASET}/bert-modify-MDL-CPI/"
    os.makedirs(dir_input, exist_ok=True)

    # with open("dataset/celegans_data.txt", "r") as f:
    with open("dataset/new_drugbank.txt", "r") as f:
        data_list = f.read().strip().split("\n")
    data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    N = len(data_list)
    print(f"总数据量 {N}")

    # Protein
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_protein_model()
    model.eval().to(device)

    #  HDF5
    h5_path = os.path.join(dir_input, "all_data.h5")
    if os.path.exists(h5_path):
        os.remove(h5_path)
    f = h5py.File(h5_path, "w")

    dt_float32 = h5py.special_dtype(vlen=np.dtype('float32'))
    dt_int64   = h5py.special_dtype(vlen=np.dtype('int64'))

    # datasets
    d_comp1 = f.create_dataset("compounds1", (N,), dtype=dt_float32)
    d_comp1_shape = f.create_dataset("compounds1_shape", (N, 2), dtype=np.int64)

    d_comp2 = f.create_dataset("compounds2", (N,), dtype=dt_int64)

    d_adj = f.create_dataset("adjacencies", (N,), dtype=dt_float32)
    d_adj_shape = f.create_dataset("adjacencies_shape", (N, 2), dtype=np.int64)

    d_prot1 = f.create_dataset("proteins1", (N,), dtype=dt_float32)
    d_prot1_shape = f.create_dataset("proteins1_shape", (N, 2), dtype=np.int64)

    d_prot2 = f.create_dataset("proteins2", (N,), dtype=dt_int64)

    d_label = f.create_dataset("interactions", (N,), dtype=np.int64)

    for idx, line in enumerate(data_list, 1):
        try:
            smiles, sequence, label = line.strip().split()

            # Protein embedding
            token_ids = np.array(tokenizer.encode(sequence), dtype=np.int64)
            token_ids_tensor = torch.tensor(token_ids[None, :], dtype=torch.long, device=device)
            input_mask_tensor = torch.ones_like(token_ids_tensor, dtype=torch.long, device=device)
            with torch.no_grad():
                output = model(token_ids_tensor, input_mask_tensor)
            protein_emb = output[0].cpu().numpy().reshape(-1, 768)

            # Compound
            atom_feat, adj = mol_features(smiles)
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            atoms = [a.GetSymbol() for a in mol.GetAtoms()]
            fingerprints = np.array([hash(x) % 10000 for x in atoms], dtype=np.int64)

            # Protein words
            words = np.array([hash(sequence[i:i+ngram]) % 10000
                             for i in range(len(sequence)-ngram+1)], dtype=np.int64)

            # HDF5
            d_comp1[idx-1] = atom_feat.flatten().astype(np.float32)
            d_comp1_shape[idx-1] = atom_feat.shape

            d_comp2[idx-1] = fingerprints

            d_adj[idx-1] = adj.flatten().astype(np.float32)
            d_adj_shape[idx-1] = adj.shape

            d_prot1[idx-1] = protein_emb.flatten().astype(np.float32)
            d_prot1_shape[idx-1] = protein_emb.shape

            d_prot2[idx-1] = words
            d_label[idx-1] = int(float(label))

        except Exception as e:
            print(f"跳过异常: {e}")
            continue

        if idx % 100 == 0:
            print(f"已处理 {idx}/{N}")
            f.flush()

    f.close()
    print(f"已保存单文件 HDF5: {h5_path}")
