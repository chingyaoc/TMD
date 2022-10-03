import os
import os.path as osp
import numpy as np
import torch
import argparse
import torch_geometric
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from tmd import TMD

parser = argparse.ArgumentParser(description='Tree Mover Distance')
parser.add_argument('--w', default=0.5, type=float, help='Layer weighting term')
parser.add_argument('--L', default=4, type=int, help='Depth of computational tree')
parser.add_argument('--dataset', default='MUTAG', type=str, help='dataset name')
parser.add_argument('--idx', default=0, type=int, help='idx for batch')
parser.add_argument('--n_per_idx', default=50, type=int, help='batch size')

# args parse
args = parser.parse_args()
w, L, dataset_name = args.w, args.L, args.dataset
n_per_idx = args.n_per_idx

path = osp.join('data', dataset_name)
train_dataset = TUDataset(path, name=dataset_name)
n = len(train_dataset)
start = n_per_idx * args.idx
end = min(n_per_idx * (args.idx + 1), n)

print('Precompute pairwise distance')
M = np.zeros((n_per_idx, n)) - 1.
for i in tqdm(range(start, end)):
    for j in tqdm(range(start, n)):
        M[i-start, j] = TMD(train_dataset[i], train_dataset[j], w=w, L=L)

if not os.path.exists('PairDist'):
    os.mkdir('PairDist')
np.save('./PairDist/M_'+dataset_name+'_L'+str(L)+'_w'+str(w)+'_idx'+str(args.idx)+'.npy', M)


