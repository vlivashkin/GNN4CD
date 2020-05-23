import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
    

# Return a list containing features gathered from multiple radius.
def aggregate_radius(radius, g, z):
    # initializing list to collect message passing result
    z_list = []
    g.ndata['z'] = z
    # pulling message from 1-hop neighbourhood
    g.update_all(fn.copy_src(src='z', out='m'), fn.sum(msg='m', out='z'))
    z_list.append(g.ndata['z'])
    for i in range(radius - 1):
        for j in range(2 ** i):
            #pulling message from 2^j neighborhood
            g.update_all(fn.copy_src(src='z', out='m'), fn.sum(msg='m', out='z'))
        z_list.append(g.ndata['z'])
    return z_list


class LGNNCore(nn.Module):
    def __init__(self, in_feats, out_feats, radius):
        super().__init__()
        self.out_feats = out_feats
        self.radius = radius

        self.linear_prev = nn.Linear(in_feats, out_feats)
        self.linear_deg = nn.Linear(in_feats, out_feats)
        self.linear_radius = nn.ModuleList([nn.Linear(in_feats, out_feats) for i in range(radius)])
        self.linear_fuse = nn.Linear(in_feats * 2, out_feats)
        self.bn = nn.BatchNorm1d(out_feats)

    def forward(self, g, feat_a, feat_b, deg, pm_pd):
        # term "prev"
        prev_proj = self.linear_prev(feat_a)
        # term "deg"
        deg_proj = self.linear_deg(deg * feat_a)

        # term "radius"
        # aggregate 2^j-hop features
        hop2j_list = aggregate_radius(self.radius, g, feat_a)
        # apply linear transformation
        hop2j_list = [linear(x) for linear, x in zip(self.linear_radius, hop2j_list)]
        radius_proj = sum(hop2j_list)

        # term "fuse"
        fuse = self.linear_fuse(torch.cat((torch.mm(pm_pd[:, :, 0], feat_b),
                                           torch.mm(pm_pd[:, :, 1], feat_b)), dim=1))

        # sum them together
        result = prev_proj + deg_proj + radius_proj + fuse

        # skip connection and batch norm
        n = self.out_feats // 2
        result = torch.cat([result[:, :n], F.relu(result[:, n:])], 1)
        result = self.bn(result)

        return result


class LGNNLayer(nn.Module):
    def __init__(self, in_feats, out_feats, radius):
        super().__init__()
        self.g_layer = LGNNCore(in_feats, out_feats, radius)
        self.lg_layer = LGNNCore(in_feats, out_feats, radius)

    def forward(self, g, lg, x, lg_x, deg_g, deg_lg, pm_pd):
        next_x = self.g_layer(g, x, lg_x, deg_g, pm_pd)
        pm_pd_y = torch.transpose(pm_pd, 0, 1)
        next_lg_x = self.lg_layer(lg, lg_x, x, deg_lg, pm_pd_y)
        return next_x, next_lg_x


class LGNN(nn.Module):
    def __init__(self, n_features, n_layers, radius, n_classes):
        super().__init__()
        feats = [1] + [n_features] * n_layers + [n_classes]
        self.linear = nn.Linear(feats[-1], n_classes)
        self.module_list = nn.ModuleList([
            LGNNLayer(m, n, radius) for m, n in zip(feats[:-1], feats[1:])
        ])

    def forward(self, g, lg, pm_pd):
        # compute the degrees
        deg_g = g.in_degrees().float().unsqueeze(1).to('cuda:0')
        deg_lg = lg.in_degrees().float().unsqueeze(1).to('cuda:0')
        
        # use degree as the input feature
        x, y = deg_g, deg_lg
        for module in self.module_list:
            x, y = module(g, lg, x, y, deg_g, deg_lg, pm_pd)
        return self.linear(x)
