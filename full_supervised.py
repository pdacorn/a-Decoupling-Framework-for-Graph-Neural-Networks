import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
from dgl import DGLGraph
from .utils import accuracy, full_load_data
from .mymodel import BBADGCN
from .mymodel import BBADJknet
from .mymodel import BBADAPPNP
from .mymodel import BBADincep
# torch.cuda.set_device(0)
import uuid
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--dataset', default='cornell')
parser.add_argument('--backbone', default='APPNP')
parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--epochs', type=int, default=1500, help='Number of epochs to train.')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate (1 - keep probability).')
parser.add_argument('--att_dropout', type=float, default=0.5, help='attention Dropout rate (1 - keep probability).')
parser.add_argument('--eps', type=float, default=0.1, help='Fixed scalar or learnable weight.')
parser.add_argument('--train_ratio', type=float, default=0.6, help='Ratio of training set')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument("--hopnum",
                        type=int,
                        help="hopnum is a hyperparameter that controls the number of blocks and the number of layers in the block")
parser.set_defaults(hopnum=[4,4,4,4])
args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
device = 'cpu'

checkpt_file = './BBAD'+uuid.uuid4().hex+'.pt'


# main loop
dur = []
los = []
loc = []
counter = 0
min_loss = 100.0
max_acc = 0.0


def train_step(model, optimizer, features, labels, idx_train):
    model.train()
    optimizer.zero_grad()
    output = model(features)
    acc_train = accuracy(output[idx_train], labels[idx_train].to(device))
    loss_train = F.nll_loss(output[idx_train], labels[idx_train].to(device))
    regloss = 0
    for name, param in model.named_parameters():
        if "layerRegular" in name:
            regloss += param
    loss = loss_train + 1e-4 * regloss
    loss.backward()
    optimizer.step()
    return loss_train, acc_train


def validate_step(model, features, labels, idx_val):
    model.eval()
    with torch.no_grad():
        output = model(features)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val].to(device))
        acc_val = accuracy(output[idx_val], labels[idx_val].to(device))
        return loss_val, acc_val


def test_step(model, features, labels, idx_test):
    model.load_state_dict(torch.load(checkpt_file))
    model.eval()
    with torch.no_grad():
        output = model(features)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test].to(device))
        acc_test = accuracy(output[idx_test], labels[idx_test].to(device))
        return loss_test, acc_test


def train(name,datastr, splitstr):
    g, nclass, features, labels, train, val, test = full_load_data(datastr, splitstr)
    features = features.to(device)
    labels = labels.to(device)
    train = train.to(device)
    test = test.to(device)
    val = val.to(device)
    g = g.to(device)
    deg = g.in_degrees().float().clamp(min=1)
    norm = torch.pow(deg, -0.5)
    g.ndata['d'] = norm
    if name == "GCN":
        model = BBADGCN(g, features.size()[1], args.hidden, args.hidden, nclass, args.dropout, args.hopnum,
                      args.att_dropout, False)
    elif name == "APPNP":
        model = BBADAPPNP(g, features.size()[1], args.hidden, args.hidden, nclass, args.dropout, args.eps, args.hopnum,
                        args.att_dropout, False)
    elif name == "JKnet":
        model = BBADJknet(g, features.size()[1], args.hidden, args.hidden, nclass, args.dropout, args.hopnum,
                        args.att_dropout, False)
    # create optimizer
    elif name == "incepGCN":
        model = BBADincep(g, features.size()[1], args.hidden, args.hidden, nclass, args.dropout, args.hopnum,
                        args.att_dropout, False)
    else:
        print("use other backbone")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    bad_counter = 0
    best = 999999999
    for epoch in range(args.epochs):
        loss_tra, acc_tra = train_step(model, optimizer, features, labels, train)
        loss_val, acc_val = validate_step(model, features, labels, val)
        if (epoch + 1) % 1 == 0:
            print('Epoch:{:04d}'.format(epoch + 1),
                  'train',
                  'loss:{:.3f}'.format(loss_tra),
                  'acc:{:.2f}'.format(acc_tra * 100),
                  '| val',
                  'loss:{:.3f}'.format(loss_val),
                  'acc:{:.2f}'.format(acc_val * 100))
        if loss_val < best:
            best = loss_val
            torch.save(model.state_dict(), checkpt_file)
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break

    acc = test_step(model, features, labels, test)[1]
    return acc * 100


t_total = time.time()
acc_list = []
for i in range(10):
    datastr = args.dataset
    splitstr = '\splits' +'\\'+ args.dataset + '_split_0.6_0.2_' + str(i) + '.npz'
    name = args.backbone
    acc_list.append(train(name,datastr, splitstr))
    print(i, ": {:.2f}".format(acc_list[-1]))
print("Train cost: {:.4f}s".format(time.time() - t_total))
print("Test acc.:{:.2f}".format(np.mean(acc_list)))
