import os.path as osp
import numpy as np
import torch
import argparse

import torch_geometric
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

parser = argparse.ArgumentParser(description='Tree Mover Distance')
parser.add_argument('--w', default=0.5, type=float, help='Layer weighting term')
parser.add_argument('--L', default=4, type=int, help='Depth of computational tree')
parser.add_argument('--dataset', default='MUTAG', type=str, help='dataset name')


# args parse
args = parser.parse_args()
w, L, dataset_name = args.w, args.L, args.dataset
path = osp.join('data', dataset_name)
dataset = TUDataset(path, name=dataset_name)

Ms = []
for idx in range((len(dataset) // 50) + 1):
    M = np.load('./PairDist/M_'+dataset_name+'_L'+str(L)+'_w'+str(w)+'_idx'+str(idx)+'.npy')
    Ms.append(M)

M = np.concatenate(Ms, axis=0)
M = M[:len(dataset)]
for i in range(len(dataset)):
    for j in range(len(dataset)):
        if M[i, j] == -1:
            M[i, j] = M[j, i]

np.save('./PairDist/M_'+dataset_name+'_L'+str(L)+'_w'+str(w)+'.npy', M)



