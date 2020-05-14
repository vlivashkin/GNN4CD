import numpy as np

import torch
from torch.autograd import Variable


def compute_operators(W, J):
    N = W.shape[0]
    d = W.sum(1)
    D = np.diag(d)
    QQ = W.copy()
    WW = np.zeros([N, N, J + 2])
    WW[:, :, 0] = np.eye(N)
    for j in range(J):
        WW[:, :, j + 1] = QQ.copy()
        QQ = np.minimum(np.dot(QQ, QQ), np.ones(QQ.shape))
    WW[:, :, J + 1] = D
    WW = np.reshape(WW, [N, N, J + 2])
    x = np.reshape(d, [N, 1])
    return WW, x


def get_Pm_Pd(W):
    N = W.shape[0]
    W = W * (np.ones([N, N]) - np.eye(N))
    M = int(W.sum()) // 2

    Pm, Pd = np.zeros([N, M * 2]), np.zeros([N, M * 2])
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
    return Pm, Pd


def get_P(W):
    Pm, Pd = get_Pm_Pd(W)
    P = np.concatenate((np.expand_dims(Pm, 2), np.expand_dims(Pd, 2)), axis=2)
    return P


def get_NB_2(W):
    Pm, Pd = get_Pm_Pd(W)
    Pf = (Pm + Pd) / 2
    Pt = (Pm - Pd) / 2
    NB = np.transpose(Pt).dot(Pf) * (1 - np.transpose(Pf).dot(Pt))
    return NB


def get_lg_inputs(W, J):
    if W.ndim == 3:
        W = W[0, :, :]
    WW, x = compute_operators(W, J)
    W_lg = get_NB_2(W)
    WW_lg, y = compute_operators(W_lg, J)
    P = get_P(W)
    x = x.astype(float)
    y = y.astype(float)
    WW = WW.astype(float)
    WW_lg = WW_lg.astype(float)
    P = P.astype(float)
    WW = Variable(torch.from_numpy(WW).unsqueeze(0), volatile=False)
    x = Variable(torch.from_numpy(x).unsqueeze(0), volatile=False)
    WW_lg = Variable(torch.from_numpy(WW_lg).unsqueeze(0), volatile=False)
    y = Variable(torch.from_numpy(y).unsqueeze(0), volatile=False)
    P = Variable(torch.from_numpy(P).unsqueeze(0), volatile=False)
    return WW, x, WW_lg, y, P


def get_gnn_inputs(W, J):
    W = W[0, :, :]
    WW, x = compute_operators(W, J)
    WW = WW.astype(float)
    WW = Variable(torch.from_numpy(WW).unsqueeze(0), volatile=False)
    x = Variable(torch.from_numpy(x).unsqueeze(0), volatile=False)
    return WW, x
