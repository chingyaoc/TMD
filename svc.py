import os.path as osp
import ot
import copy
import random
import argparse

import torch
from torch_geometric.datasets import TUDataset
from sklearn.svm import SVC
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Tree Mover Distance')
parser.add_argument('--w', default=0.5, type=float, help='Layer weighting term')
parser.add_argument('--L', default=4, type=int, help='Depth of computational tree')
parser.add_argument('--dataset', default='MUTAG', type=str, help='dataset name')
parser.add_argument('--rs', default=0, type=int, help='random seed')

# args parse
args = parser.parse_args()
w, L, dataset_name = args.w, args.L, args.dataset

random.seed(args.rs)
torch.manual_seed(args.rs)

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', dataset_name)
dataset = TUDataset(path, name=dataset_name)
M = np.load('./PairDist/M_'+dataset_name+'_L4_w'+str(w)+'.npy')

# shuffle
idx = [i for i in range(len(dataset))]
random.shuffle(idx)

n = len(dataset) // 10

idx_train = idx[n:]
idx_test = idx[:n]
train_dataset = dataset[idx_train]
test_dataset = dataset[idx_test]
M = M[idx, :]
M = M[:, idx]

y = []
for i in range(len(dataset)):
    y.append(dataset[i].y)
y = np.array(y)


# Cross Val
M_cv = M[n:, n:]
lams = [0.01, 0.05, 0.1]
best_k_count = np.zeros(len(lams))
for it in range(10):
    idx_cv = [i for i in range(len(train_dataset))]
    random.shuffle(idx_cv)
    n_cv = len(train_dataset) // 10
    idx_train_cv = idx_cv[n_cv:]
    idx_test_cv = idx_cv[:n_cv]

    for lam in lams:
        model = SVC(kernel = 'precomputed')
        model.fit(np.exp(-lam * M_cv[idx_train_cv][:, idx_train_cv]), y[idx_train][idx_train_cv])
        y_pred = model.predict(np.exp(-lam * M_cv[idx_test_cv][:, idx_train_cv]))
        acc = sum(y_pred == y[idx_train][idx_test_cv]) / len(y_pred)
        best_k_count[lams.index(lam)] += acc

best_lam = np.argmax(best_k_count)
lam = lams[best_lam]
M_ = np.exp(-lam*M)
model = SVC(kernel = 'precomputed')
model.fit(M_[n:, n:], y[idx_train])

y_pred = model.predict(M_[:n, n:])
acc = sum(y_pred == y[idx_test]) / len(y_pred)
print('{}, L: {}, w: {}, Acc: {}'.format(dataset_name, L, w, acc))

