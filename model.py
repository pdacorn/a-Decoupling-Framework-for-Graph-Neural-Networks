import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from dgl import function as fn
import numpy as np
from .Block import *
class BBADJknet(nn.Module):#jknet
    def __init__(self, g, in_dim, hidden_dim, layer_dim, out_dim, dropout, hop_num, att_dropout, concat):
        super(BBADJknet, self).__init__()
        self.g = g
        self.dropout = dropout
        self.hop_num = hop_num
        self.layer_dim = layer_dim
        self.layers = nn.ModuleList()

        for i in range(len(self.hop_num)):
            if i == 0:
                self.layers.append(JknetBlcok(hidden_dim,hidden_dim*self.hop_num[i], layer_dim, hop_num[i], dropout))
            else:
                j = 0
                sum = 1
                while j!=i:
                    sum = sum*self.hop_num[j]
                    j+=1
                nsum = sum*self.hop_num[i]
                self.layers.append(JknetBlcok(hidden_dim*sum, nsum*hidden_dim,layer_dim, hop_num[i], dropout))
        self.act = torch.nn.Sigmoid()
        self.att_drop = nn.Dropout(att_dropout)
        self.lr_att = nn.Linear(hidden_dim + layer_dim, 1)
        self.t1 = nn.Linear(in_dim, hidden_dim)
        self.cat = concat
        if not self.cat:
            self.t2 = nn.Linear(hidden_dim, out_dim)
        else:
            self.t2 = nn.Linear(hidden_dim * len(hop_num), out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.lr_att.weight, gain=1.414)
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward(self, h):
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)
        raw = h
        input_list = []
        num_node = h.shape[0]
        for i in range(len(self.hop_num)):
            o, h = self.layers[i](self.g, h)
            input_list.append(o)
        attention_scores = []
        attention_scores.append(self.act(self.lr_att(
            torch.cat([input_list[0], raw], dim=1))))
        for i in range(1, len(self.hop_num)):
            history_att = torch.cat(attention_scores[:i], dim=1)
            att = F.softmax(history_att, 1)
            history = torch.mul(input_list[0], self.att_drop(
                att[:, 0].view(num_node, 1)))
            for j in range(1, i):
                history = history + \
                          torch.mul(input_list[j], self.att_drop(
                              att[:, j].view(num_node, 1)))
            attention_scores.append(self.act(self.lr_att(
                torch.cat([history, input_list[i]], dim=1))))
        attention_scores = torch.cat(attention_scores, dim=1)
        attention_scores = F.softmax(attention_scores, 1)
        right_1 = torch.mul(input_list[0], self.att_drop(
            attention_scores[:, 0].view(num_node, 1)))
        if not self.cat:
            for i in range(1, len(self.hop_num)):
                right_1 = right_1 + \
                          torch.mul(input_list[i], self.att_drop(
                              attention_scores[:, i].view(num_node, 1)))
            right_1 = self.t2(right_1)
        else:
            for i in range(1, len(self.hop_num)):
                right_1 = torch.cat([right_1, torch.mul(input_list[i], self.att_drop(
                    attention_scores[:, i].view(num_node, 1)))], dim=1)
            right_1 = self.t2(right_1)

        return F.log_softmax(right_1, 1)
class  BBADincep(nn.Module):#incep
    def __init__(self, g, in_dim, hidden_dim, layer_dim, out_dim, dropout, hop_num, att_dropout, concat):
        super(BBADincep, self).__init__()
        self.g = g
        self.dropout = dropout
        self.hop_num = hop_num
        self.layer_dim = layer_dim
        self.layers = nn.ModuleList()

        for i in range(len(self.hop_num)):
            if i == 0:
                self.layers.append(incepBlcok(hidden_dim,hidden_dim*(self.hop_num[i]+1), layer_dim, hop_num[i], dropout))
            else:
                j = 0
                sum = 1
                while j!=i:
                    sum = sum*(self.hop_num[j]+1)
                    j+=1
                nsum = sum*(self.hop_num[i]+1)
                self.layers.append(incepBlcok(hidden_dim*sum, nsum*hidden_dim,layer_dim, hop_num[i], dropout))
        self.act = torch.nn.Sigmoid()
        self.att_drop = nn.Dropout(att_dropout)
        self.lr_att = nn.Linear(hidden_dim + layer_dim, 1)
        self.t1 = nn.Linear(in_dim, hidden_dim)
        self.cat = concat
        if not self.cat:
            self.t2 = nn.Linear(hidden_dim, out_dim)
        else:
            self.t2 = nn.Linear(hidden_dim * len(hop_num), out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.lr_att.weight, gain=1.414)
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward(self, h):
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)

        raw = h
        input_list = []
        num_node = h.shape[0]

        for i in range(len(self.hop_num)):
            o, h = self.layers[i](self.g, h)

            input_list.append(o)


        attention_scores = []
        attention_scores.append(self.act(self.lr_att(
            torch.cat([input_list[0], raw], dim=1))))
        for i in range(1, len(self.hop_num)):
            history_att = torch.cat(attention_scores[:i], dim=1)

            att = F.softmax(history_att, 1)
            history = torch.mul(input_list[0], self.att_drop(
                att[:, 0].view(num_node, 1)))
            for j in range(1, i):
                history = history + \
                          torch.mul(input_list[j], self.att_drop(
                              att[:, j].view(num_node, 1)))
            attention_scores.append(self.act(self.lr_att(
                torch.cat([history, input_list[i]], dim=1))))
        attention_scores = torch.cat(attention_scores, dim=1)
        attention_scores = F.softmax(attention_scores, 1)
        right_1 = torch.mul(input_list[0], self.att_drop(
            attention_scores[:, 0].view(num_node, 1)))
        if not self.cat:
            for i in range(1, len(self.hop_num)):
                right_1 = right_1 + \
                          torch.mul(input_list[i], self.att_drop(
                              attention_scores[:, i].view(num_node, 1)))
            right_1 = self.t2(right_1)
        else:
            for i in range(1, len(self.hop_num)):
                right_1 = torch.cat([right_1, torch.mul(input_list[i], self.att_drop(
                    attention_scores[:, i].view(num_node, 1)))], dim=1)
            right_1 = self.t2(right_1)

        return F.log_softmax(right_1, 1)

class BBADGCN(nn.Module):#gcn
    def __init__(self, g, in_dim, hidden_dim,layer_dim, out_dim, dropout, hop_num, att_dropout,concat):
        super(BBADGCN, self).__init__()
        self.g = g
        self.dropout = dropout
        self.hop_num = hop_num
        self.layer_dim = layer_dim
        self.layers = nn.ModuleList()
        for i in range(len(self.hop_num)):
            self.layers.append(GCNBlock(hidden_dim, layer_dim,hop_num[i],dropout))
        self.act = torch.nn.Sigmoid()
        self.att_drop = nn.Dropout(att_dropout)
        self.lr_att = nn.Linear(hidden_dim + hidden_dim, 1)
        self.t1 = nn.Linear(in_dim, hidden_dim)
        self.cat = concat
        if not self.cat:
            self.t2 = nn.Linear(hidden_dim, out_dim)
        else:
            self.t2 = nn.Linear(hidden_dim*len(hop_num),out_dim)
        self.reset_parameters()


    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.lr_att.weight, gain=1.414)
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward(self, h):
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)


        raw = h
        input_list = []
        num_node = h.shape[0]
        for i in range(len(self.hop_num)):
            o, h = self.layers[i](self.g, h)
            input_list.append(o)
        #right_1 = self.t2(o)

        attention_scores = []
        attention_scores.append(self.act(self.lr_att(
            torch.cat([input_list[0], raw], dim=1))))
        for i in range(1, len(self.hop_num)):
            history_att = torch.cat(attention_scores[:i], dim=1)
            att = F.softmax(history_att, 1)
            history = torch.mul(input_list[0], self.att_drop(
                att[:, 0].view(num_node, 1)))
            for j in range(1, i):
                history = history + \
                          torch.mul(input_list[j], self.att_drop(
                              att[:, j].view(num_node, 1)))

            attention_scores.append(self.act(self.lr_att(
                torch.cat([history, input_list[i]], dim=1))))
        attention_scores = torch.cat(attention_scores, dim=1)
        attention_scores = F.softmax(attention_scores, 1)
        right_1 = torch.mul(input_list[0], self.att_drop(
            attention_scores[:, 0].view(num_node, 1)))
        if not self.cat:
            for i in range(1, len(self.hop_num)):
                right_1 = right_1 + \
                          torch.mul(input_list[i], self.att_drop(
                              attention_scores[:, i].view(num_node, 1)))
            right_1 = self.t2(right_1)
        else:
            for i in range(1, len(self.hop_num)):
                
                right_1=torch.cat([right_1,torch.mul(input_list[i], self.att_drop(
                              attention_scores[:, i].view(num_node, 1)))],dim=1)
            right_1 = self.t2(right_1)


        return F.log_softmax(right_1, 1)


class BBADAPPNP(nn.Module):#appnp
    def __init__(self, g, in_dim, hidden_dim, layer_dim, out_dim, dropout, eps, hop_num, att_dropout, concat):
        super(BBADAPPNP, self).__init__()
        self.g = g
        self.eps = eps
        self.dropout = dropout
        self.hop_num = hop_num
        self.layer_dim = layer_dim
        self.layers = nn.ModuleList()
        for i in range(len(self.hop_num)):
            self.layers.append(APPNPBlock(hidden_dim, layer_dim, self.eps, hop_num[i], dropout))
        self.act = torch.nn.Sigmoid()
        self.att_drop = nn.Dropout(att_dropout)
        self.lr_att = nn.Linear(hidden_dim + hidden_dim, 1)
        self.t1 = nn.Linear(in_dim, hidden_dim)
        self.cat = concat
        if not self.cat:
            self.t2 = nn.Linear(hidden_dim, out_dim)
        else:
            self.t2 = nn.Linear(hidden_dim * len(hop_num), out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.lr_att.weight, gain=1.414)
        nn.init.xavier_normal_(self.t1.weight, gain=1.414)
        nn.init.xavier_normal_(self.t2.weight, gain=1.414)

    def forward(self, h):
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = torch.relu(self.t1(h))
        h = F.dropout(h, p=self.dropout, training=self.training)

        raw = h
        input_list = []
        num_node = h.shape[0]
        for i in range(len(self.hop_num)):
            o, h = self.layers[i](self.g, h)
            input_list.append(o)
        attention_scores = []
        attention_scores.append(self.act(self.lr_att(
            torch.cat([input_list[0], raw], dim=1))))
        for i in range(1, len(self.hop_num)):
            history_att = torch.cat(attention_scores[:i], dim=1)
            att = F.softmax(history_att, 1)
            history = torch.mul(input_list[0], self.att_drop(
                att[:, 0].view(num_node, 1)))
            for j in range(1, i):
                history = history + \
                          torch.mul(input_list[j], self.att_drop(
                              att[:, j].view(num_node, 1)))
            attention_scores.append(self.act(self.lr_att(
                torch.cat([history, input_list[i]], dim=1))))
        attention_scores = torch.cat(attention_scores, dim=1)
        attention_scores = F.softmax(attention_scores, 1)
        right_1 = torch.mul(input_list[0], self.att_drop(
            attention_scores[:, 0].view(num_node, 1)))
        if not self.cat:
            for i in range(1, len(self.hop_num)):
                right_1 = right_1 + \
                          torch.mul(input_list[i], self.att_drop(
                              attention_scores[:, i].view(num_node, 1)))
            right_1 = self.t2(right_1)
        else:
            for i in range(1, len(self.hop_num)):
                right_1 = torch.cat([right_1, torch.mul(input_list[i], self.att_drop(
                    attention_scores[:, i].view(num_node, 1)))], dim=1)
            right_1 = self.t2(right_1)

        return F.log_softmax(right_1, 1)
