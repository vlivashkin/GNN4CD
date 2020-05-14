import matplotlib
import numpy as np
import torch
from torch.autograd import Variable

matplotlib.use('Agg')


class Generator(object):
    def __init__(self, N_train=50, N_test=100, generative_model='SBM_multiclass', p_SBM=0.8, q_SBM=0.2, n_classes=2,
                 path_dataset='', num_examples_train=100, num_examples_test=10):
        self.N_train = N_train
        self.N_test = N_test
        self.generative_model = generative_model
        self.p_SBM = p_SBM
        self.q_SBM = q_SBM
        self.n_classes = n_classes
        self.path_dataset = path_dataset
        self.data_train = None
        self.data_test = None
        self.num_examples_train = num_examples_train
        self.num_examples_test = num_examples_test

    def SBM(self, p, q, N):
        W = np.zeros((N, N))

        n = N // 2

        W[:n, :n] = np.random.binomial(1, p, (n, n))
        W[n:, n:] = np.random.binomial(1, p, (N - n, N - n))
        W[:n, n:] = np.random.binomial(1, q, (n, N - n))
        W[n:, :n] = np.random.binomial(1, q, (N - n, n))
        W = W * (np.ones(N) - np.eye(N))
        W = np.maximum(W, W.transpose())

        perm = torch.randperm(N).numpy()
        blockA = perm < n
        labels = blockA * 2 - 1

        W_permed = W[perm]
        W_permed = W_permed[:, perm]
        return W_permed, labels

    def SBM_multiclass(self, p, q, N, n_classes):
        p_prime = 1 - np.sqrt(1 - p)
        q_prime = 1 - np.sqrt(1 - q)

        prob_mat = np.ones((N, N)) * q_prime

        n = N // n_classes

        for i in range(n_classes):
            prob_mat[i * n: (i + 1) * n, i * n: (i + 1) * n] = p_prime

        W = np.random.rand(N, N) < prob_mat
        W = W.astype(int)

        W = W * (np.ones(N) - np.eye(N))
        W = np.maximum(W, W.transpose())

        perm = torch.randperm(N).numpy()
        labels = (perm // n)

        W_permed = W[perm]
        W_permed = W_permed[:, perm]
        return W_permed, labels

    def sample_otf_single(self, is_training=True, cuda=True):
        N = self.N_train if is_training else self.N_test
        if self.generative_model == 'SBM':
            W, labels = self.SBM(self.p_SBM, self.q_SBM, N)
        elif self.generative_model == 'SBM_multiclass':
            W, labels = self.SBM_multiclass(self.p_SBM, self.q_SBM, N, self.n_classes)
        else:
            raise ValueError('Generative model {} not supported'.format(self.generative_model))
        labels = np.expand_dims(labels, 0)
        labels = Variable(torch.from_numpy(labels), volatile=not is_training)
        W = np.expand_dims(W, 0)
        return W, labels
