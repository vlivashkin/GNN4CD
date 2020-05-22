import networkx as nx
import numpy as np
from joblib import Parallel, delayed
from torch.utils.data import Dataset
from tqdm import tqdm

from .preprocessing import compute_operators, get_linear_graph


class SBM_Dataset(Dataset):
    """
    Simple wrapper of networkx SBM generator
    """

    def __init__(self, n, k, p_in, p_out, n_graphs=1000, mode='pre-generate', verbose=False):
        self.n = n
        self.n_graphs = n_graphs
        self.mode = mode

        self.sizes = [n // k] * k
        self.p = np.full((k, k), p_out)
        np.fill_diagonal(self.p, p_in)
        if verbose:
            print(f'sizes: {self.sizes}, p: {self.p.tolist()}')

        if mode == 'pre-generate':
            graph_range = tqdm(range(self.n_graphs)) if verbose else range(self.n_graphs)
            self.dataset = Parallel(n_jobs=-1)(delayed(self._generate_graph)() for _ in graph_range)

    def _generate_graph(self):
        G = nx.generators.community.stochastic_block_model(self.sizes, self.p)
        labels = nx.get_node_attributes(G, 'block')
        labels = np.array([labels[node] for node in G.nodes()], dtype=np.uint8)
        return G, labels

    def __len__(self):
        return self.n_graphs

    def __getitem__(self, idx):
        if self.mode == 'pre-generate':
            G, labels = self.dataset[idx]
        elif self.mode == 'on-the-fly':
            G, labels = self._generate_graph()
        else:
            raise NotImplementedError()
        return G, labels


class SBM_Dataset_GNN(SBM_Dataset):
    def __init__(self, n, k, p_in, p_out, J=3, **kwargs):
        super().__init__(n, k, p_in, p_out, **kwargs)
        self.J = J

    def __getitem__(self, idx):
        G, labels = super().__getitem__(idx)
        W = np.array(nx.adjacency_matrix(G).todense(), dtype=np.float32)
        WW, x = compute_operators(W, self.J)
        return WW, x, labels


class SBM_Dataset_LGNN(SBM_Dataset):
    def __init__(self, n, k, p_in, p_out, J=3, **kwargs):
        super().__init__(n, k, p_in, p_out, **kwargs)
        self.J = J

    def __getitem__(self, idx):
        G, labels = super().__getitem__(idx)
        W = np.array(nx.adjacency_matrix(G).todense(), dtype=np.float32)
        WW, x = compute_operators(W, self.J)
        W_lg, PmPd = get_linear_graph(W)
        WW_lg, y = compute_operators(W_lg, self.J)
        return WW, x, WW_lg, y, PmPd, labels
