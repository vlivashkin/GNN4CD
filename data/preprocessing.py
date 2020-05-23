import numpy as np
from scipy.sparse import coo_matrix

def compute_operators(W, J):
    """
    Computes W^k, k=0..J and degrees of W
    Takes NxN matrix W
    """
    deg = W.sum(axis=1)
    x = deg[:, None]
    D = np.diag(deg)

    WW = [np.eye(W.shape[0])]
    W2j = W.copy()
    for j in range(1, J + 1):
        WW.append(W2j)
        W2j = np.minimum(np.dot(W2j, W2j), np.ones(W2j.shape))
    WW.append(D)
    WW = np.stack(WW, axis=2)
    return WW, x


def get_linear_graph(W):
    """
    Computes line graph and matrices Pm, Pd
    """
    N = W.shape[0]
    W = W * (np.ones([N, N]) - np.eye(N))  # remove diagonal
    M = int(W.sum()) // 2  # 
    print('M', M)

    Pm, Pd = np.zeros([N, M * 2]), np.zeros([N, M * 2])  # assumed undirected graph
    print('Pm', Pm)
    p = 0
    for n in range(N):
        for m in range(n + 1, N):
            if W[n][m] == 1:
                Pm[n][p] = 1
                Pm[m][p] = 1
                Pm[n][p + M] = 1
                Pm[m][p + M] = 1

                Pd[n][p] = 1
                Pd[m][p] = -1
                Pd[n][p + M] = -1
                Pd[m][p + M] = 1

                p += 1

    Pf = (Pm + Pd) / 2
    Pt = (Pm - Pd) / 2
    W_lg = np.transpose(Pt).dot(Pf) * (1 - np.transpose(Pf).dot(Pt))
    PmPd = np.stack((Pm, Pd), axis=2)
    print('PmPd', PmPd)
    return W_lg, PmPd  # (coo_matrix(Pm), coo_matrix(Pd))
