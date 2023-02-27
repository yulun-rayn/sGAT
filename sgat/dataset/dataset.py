import os
import time
import pickle
import logging

from rdkit import Chem

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import torch_geometric as pyg
from torch_geometric.data import Data, Batch

from ..utils.graph_utils import mol_to_pyg_graph

def my_collate(samples):
    y = [s[0] for s in samples]
    g1 = [s[1] for s in samples]
    g2 = [s[2] for s in samples]

    y = torch.cat(y, dim=0)
    G1 = Batch.from_data_list(g1)
    if None in g2:
        return y, G1, None
    else:
        G2 = Batch.from_data_list(g2)
        return y, G1, G2

def parse_raw_data(raw_dataset):
    batch_size = 32
    loader = DataLoader(raw_dataset,
                        shuffle=False,
                        collate_fn=my_collate,
                        batch_size=batch_size,
                        num_workers=16)
    all_data = []
    print("\nPreprocessing {} samples".format(len(raw_dataset)))
    for i, d in enumerate(loader):
        if (i % 3)==0:
            print("{:7d}".format(i*batch_size))
            print(len(all_data))
        all_data.append(d)
        if i==20:
            break
    return all_data

def parse_data(raw_dataset, storage_path, dataset_name):
    dataset_path = os.path.join(storage_path, dataset_name+'.pkl')
    print(dataset_path)
    try:
        with open(dataset_path, 'rb') as f:
            parsed_data = pickle.load(f)
        print("Preprocessed {} set loaded".format(dataset_name))
    except Exception as e:
        print("Preprocessed {} set not found. Parsing...".format(dataset_name))
        t0 = time.time()
        parsed_data = parse_raw_data(raw_dataset)
        print("{:5.2f}s for {} samples".format(time.time()-t0, len(parsed_data)))
        with open(dataset_path, 'wb') as f:
            pickle.dump(parsed_data, f)
        print("Done.")
    return parsed_data

def parse_data_path(data_path, use_3d):
    path_split = data_path.split('/')
    parent_path = '/'.join(path_split[:-1])
    data_name = path_split[-1].split('.')[0]
    storage_path = os.path.join(parent_path, data_name)

    if use_3d:
        storage_path += '_with_3d'
    else:
        storage_path += '_no_3d'

    os.makedirs(storage_path, exist_ok=True)
    return storage_path


#############################################
#                   DATA                    #
#############################################

def get_dense_edges(n):
    x = np.arange(n)
    src, dst = [np.tile(x, len(x)), np.repeat(x, len(x))]
    return torch.tensor([src, dst], dtype=torch.long)


class MolData(Dataset):
    def __init__(self, score, smiles, use_3d):
        super(MolData, self).__init__()
        self.score = score
        self.smiles = smiles
        self.use_3d = use_3d

    def __getitem__(self, index):
        score = self.score[index]
        smiles = self.smiles[index]
        # Hot fix, get first in list if mol is none...
        mol = Chem.MolFromSmiles(smiles)

        if mol is None:
            smiles = self.smiles[0]
            score = self.score[0]
            mol = Chem.MolFromSmiles(smiles)
            print("Invalid SMILE encountered. Using first row instead.")

        g = mol_to_pyg_graph(mol, self.use_3d)
        return torch.FloatTensor([score]), g[0], g[1]

    def __len__(self):
        return len(self.score)

    def get_graph_spec(self):
        y, g, _ = self[0]
        nb_node_feats = g.x.shape[1]
        try:
            nb_edge_feats = g.edge_attr.shape[1]
        except Exception as e:
            nb_edge_feats = 1
        return nb_node_feats, nb_edge_feats

    def compute_baseline_error(self):
        score = np.array(self.score)
        mean = score.mean()
        sq_sum = np.sum(np.square(score - mean)) / len(score)
        logging.info("{:5.3f} baseline L2 loss\n".format(sq_sum))


def create_datasets(score, smiles, use_3d, np_seed=0):
    nb_samples = len(score)
    assert nb_samples > 10

    nb_train = int(nb_samples * 0.6)
    nb_valid = int(nb_samples * 0.2)

    np.random.seed(np_seed)
    sample_order = np.random.permutation(nb_samples)

    score = np.asarray(score)[sample_order].tolist()
    smiles = np.asarray(smiles)[sample_order].tolist()

    train_data = MolData(score[:nb_train], smiles[:nb_train], use_3d)
    valid_data = MolData(score[nb_train:nb_train + nb_valid],
                         smiles[nb_train:nb_train + nb_valid],
                         use_3d)
    test_data = MolData(score[nb_train + nb_valid:],
                        smiles[nb_train + nb_valid:],
                        use_3d)
    return train_data, valid_data, test_data
