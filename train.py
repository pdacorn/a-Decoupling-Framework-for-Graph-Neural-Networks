
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from .utils import accuracy, preprocess_data
from .mymodel import BBADGCN
from .mymodel import BBADJknet
from .mymodel import BBADAPPNP
from .mymodel import BBADincep
# torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--dataset', default='cora')
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

g, nclass, features, labels, train, val, test = preprocess_data(args.dataset, args.train_ratio)
features = features.to(device)
labels = labels.to(device)
train = train.to(device)
test = test.to(device)
val = val.to(device)

g = g.to(device)
deg = g.in_degrees().float().clamp(min=1)
norm = torch.pow(deg, -0.5)
g.ndata['d'] = norm
if args.backbone == "GCN":
    net = BBADGCN(g, features.size()[1], args.hidden,args.hidden,nclass, args.dropout, args.hopnum,args.att_dropout,False)
elif args.backbone == "APPNP":
    net = BBADAPPNP(g, features.size()[1], args.hidden,args.hidden,nclass, args.dropout, args.eps, args.hopnum,args.att_dropout,False)
elif args.backbone == "JKnet":
    net = BBADJknet(g, features.size()[1], args.hidden, args.hidden, nclass, args.dropout, args.hopnum,args.att_dropout,False)

elif args.backbone == "incepGCN":
    net = BBADincep(g, features.size()[1], args.hidden, args.hidden, nclass, args.dropout, args.hopnum,args.att_dropout,False)
else:
    print("use other backbone")
    # create optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)




# main loop
dur = []
los = []
loc = []
counter = 0
min_loss = 100.0
max_acc = 0.0

for epoch in range(args.epochs):
    if epoch >= 3:
        t0 = time.time()

    net.train()
    logp = net(features)
    cla_loss = F.nll_loss(logp[train], labels[train])
    regloss = 0

    for name, param in net.named_parameters():
        if "layerRegular" in name:
            regloss += param


    loss = cla_loss+1e-4*regloss

    train_acc = accuracy(logp[train], labels[train])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    net.eval()
    logp = net(features)
    test_acc = accuracy(logp[test], labels[test])
    loss_val = F.nll_loss(logp[val], labels[val]).item()
    val_acc = accuracy(logp[val], labels[val])
    los.append([epoch, loss_val, val_acc, test_acc])

    if loss_val < min_loss and max_acc < val_acc:
        min_loss = loss_val
        max_acc = val_acc
        counter = 0
    else:
        counter += 1

    if counter >= args.patience and args.dataset in ['cora', 'citeseer', 'pubmed']:
        print('early stop')
        break

    if epoch >= 3:
        dur.append(time.time() - t0)

    print("Epoch {:05d} | Loss {:.4f} | Train {:.4f} | Val {:.4f} | Test {:.4f} | Time(s) {:.4f}".format(
        epoch, loss_val, train_acc, val_acc, test_acc, np.mean(dur)))

if args.dataset in ['cora', 'citeseer', 'pubmed'] or 'syn' in args.dataset:
    los.sort(key=lambda x: x[1])
    acc = los[0][-1]
    print(los[0][0],acc)
else:
    los.sort(key=lambda x: -x[2])
    acc = los[0][-1]
    print(acc)

