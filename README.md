# Tree Mover's Distance for Graphs

<p align='center'>
<img src='https://github.com/chingyaoc/TMD/blob/master/misc/fig.png?raw=true' width='900'/>
</p>


Understanding generalization and robustness of machine learning models fundamentally relies on assuming an appropriate metric on the data space. Identifying such a metric is particularly challenging for non-Euclidean data such as graphs. Here, we propose a pseudometric for attributed graphs, the Tree Mover's Distance (TMD), and study its relation to generalization. Via a hierarchical optimal transport problem, TMD reflects the local distribution of node attributes as well as the distribution of local computation trees, which are known to be decisive for the learning behavior of graph neural networks (GNNs). First, we show that TMD captures properties relevant to graph classification: a simple TMD-SVM performs competitively with standard GNNs. Second, we relate TMD to generalization of GNNs under distribution shifts, and show that it correlates well with performance drop under such shifts.

**Tree Mover’s Distance: Bridging Graph Metrics and Stability of Graph Neural Networks** NeurIPS 2022 [[paper]](https://arxiv.org/abs/2210.01906)
<br/>
[Ching-Yao Chuang](https://chingyaoc.github.io/) and 
[Stefanie Jegelka](https://people.csail.mit.edu/stefje/)
<br/>


## Prerequisites
- Python 3.7 
- PyTorch 1.3.1
- PyTorch Geometric
- POT


## Usage Examples
The code for computing Tree Mover's Distance (TMD) lie in `tmd.py`. For instance, the following code compute the TMD between two graphs of MUTAG dataset.
```python
from tmd import TMD
from torch_geometric.datasets import TUDataset

dataset = TUDataset('data', name='MUTAG')
d = TMD(dataset[0], dataset[1], w=1.0, L=4)
```

One can also specify different weighting constants for each layer as follows:
```python
d = TMD(dataset[0], dataset[1], w=[0.33, 1, 3], L=4)
```
This results in a tighter bound on the stability of GNNs as Theorem 8 shows. Note that `len(w)` has to be the same as `L-1`.


## Graph Classification on TUDataset

Step 1: Pre-compute the pairwise distance (potentially parallel). For instance, the following script compute the pairwise distances of MUTAG with `pairwise_dist.py` by separating it into 4 batches, where each batch is computed parallely. One can merge the batches with `merge.py`.
```
python pairwise_dist.py --w 0.5 --L 4 --dataset MUTAG --n_per_idx 50 --idx 0
python pairwise_dist.py --w 0.5 --L 4 --dataset MUTAG --n_per_idx 50 --idx 1
python pairwise_dist.py --w 0.5 --L 4 --dataset MUTAG --n_per_idx 50 --idx 2
python pairwise_dist.py --w 0.5 --L 4 --dataset MUTAG --n_per_idx 50 --idx 3

python merge.py --w 0.5 --L 4 --dataset MUTAG
```

Step 2: Train a SVM classifier based on the pre-computed distances:
```
python svc.py --w 0.5 --L 4 --dataset MUTAG
```

## Measuring the Stability of GNNs
The script `stability.py` reproduce the stability experiments in Figure 5. In particular, it plots the correlation between a (L+1)-layer GIN and the tree mover's distance with graphs sampled from MUTAG.
```
python stability.py --L 3
```

<p align='left'>
<img src='https://github.com/chingyaoc/TMD/blob/master/misc/fig_stable.png?raw=true' width='700'/>
</p>


## Citation

If you find this repo useful for your research, please consider citing the paper

```
@article{chuang2022tree,
  title={Tree Mover’s Distance: Bridging Graph Metrics and Stability of Graph Neural Networks},
  author={Chuang, Ching-Yao and Jegelka, Stefanie},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  year={2022}
}
```


