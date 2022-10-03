import random
import numpy as np
import os.path as osp
from tqdm import tqdm
from scipy.stats import pearsonr

import torch
import torch.nn.functional as F
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential, BCEWithLogitsLoss
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_add_pool

from tmd import TMD

random.seed(1)
torch.manual_seed(1)

# TU Dataset
dataset = TUDataset('data', name='MUTAG').shuffle()
train_dataset = dataset[len(dataset) // 10:]
test_dataset = dataset[:len(dataset) // 10]
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)


class Net(torch.nn.Module):
    '''
    3-layer GIN Network
    '''
    def __init__(self, in_channels, dim, out_channels):
        super().__init__()
        self.conv1 = GINConv(
            Sequential(Linear(in_channels, dim), ReLU(),
                       Linear(dim, dim), ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(dim, dim), ReLU(),
                       Linear(dim, dim), ReLU()))
        self.conv3 = GINConv(
            Sequential(Linear(dim, dim), ReLU(),
                       Linear(dim, dim), ReLU()))
        self.lin1 = Linear(dim, dim)
        self.lin2 = Linear(dim, 1)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        x = global_add_pool(x, batch)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net(dataset.num_features, 32, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = BCEWithLogitsLoss()

def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data.x, data.edge_index, data.batch)
        loss = criterion(output[:,0], data.y.float())
        loss.backward()
        optimizer.step()
        total_loss += float(loss) * data.num_graphs
    return total_loss / len(train_loader.dataset)


@torch.no_grad()
def test(loader):
    model.eval()

    total_correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        total_correct += int(((F.sigmoid(out[:, 0]) > 0.5).int() == data.y).sum())
    return total_correct / len(loader.dataset)


for epoch in range(1, 101):
    loss = train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f} '
          f'Test Acc: {test_acc:.4f}')


oo = []
tt = []
for i in tqdm(range(1000)):
    a = random.randint(0, len(dataset)-1)
    b = random.randint(0, len(dataset)-1)
    g_a = dataset[a].cuda()
    g_b = dataset[b].cuda()

    # output from GIN
    output_a = model(g_a.x, g_a.edge_index, torch.zeros(len(g_a.x), dtype=torch.int64).cuda())
    output_b = model(g_b.x, g_b.edge_index, torch.zeros(len(g_b.x), dtype=torch.int64).cuda())

    # TMD
    tmd = TMD(dataset[a], dataset[b], w=[1/3, 1, 3], L=4)

    oo.append(float(torch.norm(output_a - output_b).cpu().detach().numpy()))
    tt.append(tmd)
    

import matplotlib.pyplot as plt
plt.scatter(oo, tt)
plt.savefig('gnn_plot.png', dpi=120)
print('Pearson correlation: {}'.format(pearsonr(np.array(oo), np.array(tt))))
