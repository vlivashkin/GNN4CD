import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn

from data_generator import Generator
from load import get_gnn_inputs
from losses import compute_loss_multiclass, compute_accuracy_multiclass
from models import GNN_multiclass, lGNN_multiclass

if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    dtype_l = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    dtype_l = torch.LongTensor

criterion = nn.CrossEntropyLoss()


def train_mcd_single(gnn: GNN_multiclass, optimizer, gen: Generator, n_classes, it):
    start = time.time()
    W, labels = gen.sample_otf_single(is_training=True, cuda=torch.cuda.is_available())
    labels = labels.type(dtype_l)

    if args.generative_model == 'SBM_multiclass' and args.n_classes == 2:
        labels = (labels + 1) / 2

    WW, x = get_gnn_inputs(W, args.J)

    if torch.cuda.is_available():
        WW.cuda()
        x.cuda()

    pred = gnn(WW.type(dtype), x.type(dtype))

    loss = compute_loss_multiclass(pred, labels, n_classes)
    gnn.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm(gnn.parameters(), args.clip_grad_norm)
    optimizer.step()

    acc = compute_accuracy_multiclass(pred, labels, n_classes)

    elapsed = time.time() - start

    if torch.cuda.is_available():
        loss_value = float(loss.data.cpu().numpy())
    else:
        loss_value = float(loss.data.numpy())

    # print(f"{'iter':<10} {'avg loss':<10} {'avg acc':<10} {'model':<10} {'elapsed':<10} ")
    print(f"{it:<10} {loss_value:<10.5f} {acc:<10.5f} {'GNN':<10} {elapsed:<10.3f}")

    del WW
    del x

    return loss_value, acc


def train(gnn: GNN_multiclass, gen: Generator, n_classes, iters):
    gnn.train()
    optimizer = torch.optim.Adamax(gnn.parameters(), lr=args.lr)
    loss_lst = np.zeros([iters])
    acc_lst = np.zeros([iters])
    for it in range(iters):
        loss_single, acc_single = train_mcd_single(gnn, optimizer, gen, n_classes, it)
        loss_lst[it] = loss_single
        acc_lst[it] = acc_single
        torch.cuda.empty_cache()
    print('Avg train loss', np.mean(loss_lst))
    print('Avg train acc', np.mean(acc_lst))
    print('Std train acc', np.std(acc_lst))


def test_mcd_single(gnn: GNN_multiclass, gen: Generator, n_classes, it):
    start = time.time()
    W, labels = gen.sample_otf_single(is_training=False, cuda=torch.cuda.is_available())
    labels = labels.type(dtype_l)
    if args.generative_model == 'SBM_multiclass' and args.n_classes == 2:
        labels = (labels + 1) / 2
    WW, x = get_gnn_inputs(W, args.J)

    print('WW', WW.shape)

    if torch.cuda.is_available():
        WW.cuda()
        x.cuda()

    pred_single = gnn(WW.type(dtype), x.type(dtype))
    labels_single = labels

    loss_test = compute_loss_multiclass(pred_single, labels_single, n_classes)
    acc_test = compute_accuracy_multiclass(pred_single, labels_single, n_classes)

    elapsed = time.time() - start

    if torch.cuda.is_available():
        loss_value = float(loss_test.data.cpu().numpy())
    else:
        loss_value = float(loss_test.data.numpy())

    # print(f"{'iter':<10} {'avg loss':<10} {'avg acc':<10} {'model':<10} {'elapsed':<10} ")
    print(f"{it:<10} {loss_value:<10.5f} {acc_test:<10.5f} {'GNN':<10} {elapsed:<10.3f}")

    del WW
    del x

    return loss_value, acc_test


def test(gnn, gen: Generator, n_classes, iters):
    gnn.train()
    loss_lst = np.zeros([iters])
    acc_lst = np.zeros([iters])
    for it in range(iters):
        loss_single, acc_single = test_mcd_single(gnn, gen, n_classes, it)
        loss_lst[it] = loss_single
        acc_lst[it] = acc_single
        torch.cuda.empty_cache()
    print('Avg test loss', np.mean(loss_lst))
    print('Avg test acc', np.mean(acc_lst))
    print('Std test acc', np.std(acc_lst))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    ###############################################################################
    #                             General Settings                                #
    ###############################################################################

    parser.add_argument('--num_examples_train', nargs='?', const=1, type=int, default=6000)
    parser.add_argument('--num_examples_test', nargs='?', const=1, type=int, default=100)
    parser.add_argument('--p_SBM', nargs='?', const=1, type=float, default=0.3)
    parser.add_argument('--q_SBM', nargs='?', const=1, type=float, default=0.15)
    parser.add_argument('--generative_model', nargs='?', const=1, type=str, default='SBM_multiclass')
    parser.add_argument('--batch_size', nargs='?', const=1, type=int, default=1)
    parser.add_argument('--mode', nargs='?', const=1, type=str, default='train')
    parser.add_argument('--path_gnn', nargs='?', const=1, type=str, default='')
    parser.add_argument('--filename_existing_gnn', nargs='?', const=1, type=str, default='')
    parser.add_argument('--print_freq', nargs='?', const=1, type=int, default=10)
    parser.add_argument('--test_freq', nargs='?', const=1, type=int, default=500)
    parser.add_argument('--save_freq', nargs='?', const=1, type=int, default=2000)
    parser.add_argument('--clip_grad_norm', nargs='?', const=1, type=float, default=40.0)
    parser.add_argument('--freeze_bn', dest='eval_vs_train', action='store_true')
    parser.set_defaults(eval_vs_train=False)

    ###############################################################################
    #                                 GNN Settings                                #
    ###############################################################################

    parser.add_argument('--num_features', nargs='?', const=1, type=int, default=8)
    parser.add_argument('--num_layers', nargs='?', const=1, type=int, default=30)
    parser.add_argument('--n_classes', nargs='?', const=1, type=int, default=5)
    parser.add_argument('--J', nargs='?', const=1, type=int, default=2)
    parser.add_argument('--N_train', nargs='?', const=1, type=int, default=400)
    parser.add_argument('--N_test', nargs='?', const=1, type=int, default=400)
    parser.add_argument('--lr', nargs='?', const=1, type=float, default=0.004)

    args = parser.parse_args()
    print(args)

    batch_size = args.batch_size

    # One fixed generator
    gen = Generator(N_train=args.N_train, N_test=args.N_test, p_SBM=args.p_SBM, q_SBM=args.q_SBM,
                    generative_model=args.generative_model, n_classes=args.n_classes)

    torch.backends.cudnn.enabled = False

    if args.mode == 'test':
        print('In testing mode')
        filename = args.filename_existing_gnn
        path_plus_name = os.path.join(args.path_gnn, filename)
        if (filename != '') and (os.path.exists(path_plus_name)):
            print('Loading gnn ' + filename)
            gnn = torch.load(path_plus_name)
            if torch.cuda.is_available():
                gnn.cuda()
        else:
            print('No such a gnn exists; creating a brand new one')
            if args.generative_model == 'SBM':
                gnn = lGNN_multiclass(args.num_features, args.num_layers, args.J + 2)
            elif args.generative_model == 'SBM_multiclass':
                gnn = lGNN_multiclass(args.num_features, args.num_layers, args.J + 2, n_classes=args.n_classes)
            filename = 'gnn_J' + str(args.J) + '_lyr' + str(args.num_layers) + '_Ntr' + str(
                args.N_train) + '_num' + str(args.num_examples_train)
            path_plus_name = os.path.join(args.path_gnn, filename)
            if torch.cuda.is_available():
                gnn.cuda()
            print('Training begins')
    elif args.mode == 'train':
        filename = args.filename_existing_gnn
        path_plus_name = os.path.join(args.path_gnn, filename)
        if (filename != '') and (os.path.exists(path_plus_name)):
            print('Loading gnn ' + filename)
            gnn = torch.load(path_plus_name)
            filename = filename + '_Ntr' + str(args.N_train) + '_num' + str(args.num_examples_train)
            path_plus_name = os.path.join(args.path_gnn, filename)
        else:
            print('No such a gnn exists; creating a brand new one')
            filename = 'gnn_J' + str(args.J) + '_lyr' + str(args.num_layers) + '_Ntr' + str(args.N_train) + \
                       '_num' + str(args.num_examples_train)
            path_plus_name = os.path.join(args.path_gnn, filename)
            if args.generative_model == 'SBM':
                gnn = GNN_multiclass(args.num_features, args.num_layers, args.J + 2, n_classes=2)
            elif args.generative_model == 'SBM_multiclass':
                gnn = GNN_multiclass(args.num_features, args.num_layers, args.J + 2, n_classes=args.n_classes)

        print('total num of params:', count_parameters(gnn))

        if torch.cuda.is_available():
            gnn.cuda()
        print('Training begins')
        if args.generative_model == 'SBM':
            train(gnn, gen, 2, args.num_examples_train)
        elif args.generative_model == 'SBM_multiclass':
            train(gnn, gen, args.n_classes, args.num_examples_train)
        print('Saving gnn ' + filename)
        if torch.cuda.is_available():
            torch.save(gnn.cpu(), path_plus_name)
            gnn.cuda()
        else:
            torch.save(gnn, path_plus_name)

    print('Testing the GNN:')
    if args.eval_vs_train:
        print('model status: eval')
        gnn.eval()
    else:
        print('model status: train')
        gnn.train()

    test(gnn, gen, args.n_classes, args.num_examples_train)

    print('total num of params:', count_parameters(gnn))
