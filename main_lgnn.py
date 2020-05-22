from time import gmtime, strftime

import numpy as np
import torch
import torch.nn as nn
from munch import munchify
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.sbm import SBM_Dataset_LGNN
from models.lgnn import LGNN
from models.losses import combinatorical_accuracy, ari_score, combinatorical_cce

device = 'cpu'  # 'cuda:0'
args = munchify({
    'clip_grad_norm': 40.0,
    'num_features': 8,
    'num_layers': 30,
    'n_classes': 5,
    'J': 2,
    'lr': 0.004
})

n, k, p_in, p_out = 50, args.n_classes, 0.8, 0.2
n_epoch, n_samples_train, n_samples_test = 3, 500, 100
train_dataset = SBM_Dataset_LGNN(n, k, p_in, p_out, J=args.J, n_graphs=n_samples_train)
test_dataset = SBM_Dataset_LGNN(n, k, p_in, p_out, J=args.J, n_graphs=n_samples_test)
train_dataloader = DataLoader(train_dataset, batch_size=1)
test_dataloader = DataLoader(test_dataset, batch_size=1)

torch.backends.cudnn.enabled = False
model = LGNN(args.num_features, args.num_layers, args.J + 2, n_classes=args.n_classes).to(device)
optimizer = torch.optim.Adamax(model.parameters(), lr=args.lr)

name = f'GNN4CD_LGNN-SBM({n}, {k}, {p_in:.2f}, {p_out:.2f})'
writer = SummaryWriter(f'./logs/{strftime("%Y-%m-%d %H:%M:%S", gmtime())} {name}')
for epoch in range(n_epoch):
    model.train()
    for it, (WW, x, WW_lg, y, P, labels) in enumerate(tqdm(train_dataloader, desc=str(epoch))):
        WW, x, WW_lg, y, P = [x.float().to(device) for x in (WW, x, WW_lg, y, P)]
        labels = labels.long().to(device)
        pred = model(WW, x, WW_lg, y, P)

        loss = combinatorical_cce(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
        optimizer.step()

        writer.add_scalar('train/loss', loss.item(), epoch * len(train_dataloader) + it)
        writer.add_scalar('train/acc', combinatorical_accuracy(pred, labels), epoch * len(train_dataloader) + it)
        writer.add_scalar('train/ari', ari_score(pred, labels), epoch * len(train_dataloader) + it)
        if it % 100 == 0:
            writer.flush()

    loss_lst, acc_lst, ari_lst = [], [], []
    with torch.no_grad():
        for _, (WW, x, WW_lg, y, P, labels) in enumerate(test_dataloader):
            WW, x, WW_lg, y, P, labels = [x.float().to(device) for x in (WW, x, WW_lg, y, P, labels)]
            pred = model(WW, x, WW_lg, y, P)

            loss = combinatorical_cce(pred, labels)
            loss_lst.append(loss.item())
            acc_lst.append(combinatorical_accuracy(pred, labels))
            ari_lst.append(ari_score(pred, labels))
    writer.add_scalar('test/loss', np.mean(loss_lst), epoch)
    writer.add_scalar('test/acc', np.mean(acc_lst), epoch)
    writer.add_scalar('test/ari', np.mean(ari_lst), epoch)
    writer.flush()
