# Tree Mover's Distance


**Tree Mover’s Distance: Bridging Graph Metrics and Stability of Graph Neural Networks** NeurIPS 2022 [[paper]](https://arxiv.org/abs/2007.00224)
<br/>
[Ching-Yao Chuang](https://chingyaoc.github.io/) and 
[Stefanie Jegelka](https://people.csail.mit.edu/stefje/)
<br/>


## Prerequisites
- Python 3.7 
- PyTorch 1.3.1
- PyTorch Geometric
- POT


## USAGE
The code for computing Tree Mover's Distance (TMD) lie in `tmd.py`. For instance, the following code compute the TMD between two graphs of MUTAG dataset.
```python
from tmd import TMD
from torch_geometric.datasets import TUDataset

dataset = TUDataset('data', name='MUTAG')
d = TMD(dataset[0], dataset[1], w=1.0, L=4)
```


## Graph Classification on TUDataset

Pre-compute the pairwise distance (potentially parallel). This generates the pair-wise distance matrix in `PairDist/M_MUTAG_L4_w0.5.npy`.
```
python pairwise_dist.py --w 0.5 --L 4 --dataset MUTAG --idx 0
python pairwise_dist.py --w 0.5 --L 4 --dataset MUTAG --idx 1
python pairwise_dist.py --w 0.5 --L 4 --dataset MUTAG --idx 2
python pairwise_dist.py --w 0.5 --L 4 --dataset MUTAG --idx 3

python merge.py --w 0.5 --L 4 --dataset MUTAG
```

Train a SVM classifier based on the tree mover's distance:
```
python svc.py --w 0.5 --L 4 --dataset MUTAG
```

## Measure Stability of GNNs



