import torch
import torch.nn as nn
import torch.nn.functional as F


def GMul(W, x):
    """
        W is a tensor of size (bs, N, N, J)
        x is a tensor of size (bs, N, num_features)
    """
    N = W.shape[1]
    W_lst = W.split(1, 3)
    W = torch.cat(W_lst, 1).squeeze(3)  # W is now a tensor of size (bs, J*N, N)
    output = torch.bmm(W, x)  # output has size (bs, J*N, num_features)
    output = output.split(N, 1)
    output = torch.cat(output, 2)  # output has size (bs, N, J*num_features)
    return output


class GNNAtomic(nn.Module):
    def __init__(self, feature_maps, J):
        super(GNNAtomic, self).__init__()
        self.num_inputs = J * feature_maps[0]
        self.num_outputs = feature_maps[2]
        self.fc1 = nn.Linear(self.num_inputs, self.num_outputs // 2)
        self.fc2 = nn.Linear(self.num_inputs, self.num_outputs - self.num_outputs // 2)
        self.bn2d = nn.BatchNorm1d(self.num_outputs)

    def forward(self, WW, x):
        x = GMul(WW, x)
        x_size = x.size()
        x = x.contiguous()
        x = x.view(-1, self.num_inputs)
        x1 = F.relu(self.fc1(x))  # has size (bs*N, num_outputs)
        x2 = self.fc2(x)
        x = torch.cat((x1, x2), 1)
        x = self.bn2d(x)
        x = x.view(*x_size[:-1], self.num_outputs)
        return x


class GNNAtomicLast(nn.Module):
    def __init__(self, feature_maps, J, n_classes):
        super(GNNAtomicLast, self).__init__()
        self.num_inputs = J * feature_maps[0]
        self.num_outputs = n_classes
        self.fc = nn.Linear(self.num_inputs, self.num_outputs)

    def forward(self, WW, x):
        x = GMul(WW, x)  # out has size (bs, N, num_inputs)
        x_size = x.size()
        x = x.contiguous()
        x = x.view(x_size[0] * x_size[1], -1)
        x = self.fc(x)  # has size (bs*N, num_outputs)
        x = x.view(*x_size[:-1], self.num_outputs)
        return x


class GNN(nn.Module):
    def __init__(self, num_features, num_layers, J, n_classes=2):
        super(GNN, self).__init__()
        self.num_features = num_features
        self.num_layers = num_layers
        self.featuremap_in = [1, 1, num_features]
        self.featuremap_mi = [num_features, num_features, num_features]
        self.featuremap_end = [num_features, num_features, num_features]
        self.layer0 = GNNAtomic(self.featuremap_in, J)
        for i in range(num_layers):
            module = GNNAtomic(self.featuremap_mi, J)
            self.add_module('layer{}'.format(i + 1), module)
        self.layerlast = GNNAtomicLast(self.featuremap_end, J, n_classes)

    def forward(self, W, x):
        x = self.layer0(W, x)
        for i in range(self.num_layers):
            x = self._modules['layer{}'.format(i + 1)](W, x)
        out = self.layerlast(W, x)
        return out[1]
