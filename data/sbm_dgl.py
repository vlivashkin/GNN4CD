import networkx as nx
import numpy as np
import dgl
import torch

from .preprocessing import compute_operators, get_linear_graph
from .sbm import SBM_Dataset

class SBM_Dataset_LGNN_DGL(SBM_Dataset):
    def _generate_graph(self):
        G, labels = super()._generate_graph()
        W = np.array(nx.adjacency_matrix(G).todense(), dtype=np.float32)
        W_lg, PmPd = get_linear_graph(W)
        G, G_lg = dgl.DGLGraph(W), dgl.DGLGraph(W_lg)
        return G, G_lg, PmPd, labels

    def collate_fn(self, x):
        G, G_lg, PmPd, labels = zip(*x)
        return G[0], G_lg[0], torch.from_numpy(PmPd[0]).float(), torch.from_numpy(labels[0]).long()