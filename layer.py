import torch
from torch import nn
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.function as fn
from torch.nn.parameter import Parameter


class APPNPLayer(nn.Module):#APPNP
    def __init__(self,in_dim,hop_num,fea_drop,alpha):
        super(APPNPLayer, self).__init__()
        self.gate = nn.Linear(2 * in_dim, 1)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)
        self.hop_num = hop_num
        self.alpha = alpha
        self.feature_drop = nn.Dropout(fea_drop)
        self.layerRegular = []
        for i in range(self.hop_num):
            self.layerRegular.append(Parameter(torch.rand(1,)))
            self.register_parameter('layerRegular'+str(i),self.layerRegular[i])
    def edge_applying(self, edges):
        g = torch.ones(len(edges),1).squeeze()
        e = g * edges.dst['d'] * edges.src['d']
        return {'e': e, 'm': g}
    def forward(self, graph: DGLGraph,h):
        graph = graph.local_var()
        graph.ndata['h'] = h
        graph.apply_edges(self.edge_applying)
        feat = h
        degree_normal = graph.edata.pop('e')
        for _ in range(self.hop_num):
            graph.ndata['h'] = feat
            graph.edata['e'] = degree_normal
            graph.update_all(fn.u_mul_e('h', 'e', '_'), fn.sum('_', 'h'))
            feat_p = graph.ndata.pop('h') *(1-self.alpha)+ self.alpha *h
            feat = feat_p * self.layerRegular[_] + (1 - self.layerRegular[_]) * feat
            feat = self.feature_drop(feat)
        return feat
class GCNLayer(nn.Module):#GCN
    def __init__(self,in_dim,hop_num,fea_drop):
        super(GCNLayer, self).__init__()
        self.gate = nn.Linear(2 * in_dim, 1)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)
        self.hop_num = hop_num
        self.feature_drop = nn.Dropout(fea_drop)
        self.layerRegular = []
        for i in range(self.hop_num):
            self.layerRegular.append(Parameter(torch.rand(1,)))
            self.register_parameter('layerRegular'+str(i),self.layerRegular[i])
    def edge_applying(self, edges):
        g = torch.ones(len(edges),1).squeeze()
        e = g * edges.dst['d'] * edges.src['d']
        return {'e': e, 'm': g}
    def forward(self, graph: DGLGraph,h):
        graph = graph.local_var()
        graph.ndata['h'] = h
        graph.apply_edges(self.edge_applying)
        feat = h
        degree_Nomal = graph.edata.pop('e')
        for _ in range(self.hop_num):
            graph.ndata['h'] = feat
            graph.edata['e'] = degree_Nomal
            graph.update_all(fn.u_mul_e('h', 'e', '_'), fn.sum('_', 'h'))
            feat = graph.ndata.pop('h') * self.layerRegular[_] + (1 - self.layerRegular[_]) * feat
            feat = self.feature_drop(feat)
        return feat



class JKnetLayer(nn.Module):#jknet
    def __init__(self,in_dim,hop_num,fea_drop):
        super(JKnetLayer, self).__init__()
        self.gate = nn.Linear(2 * in_dim, 1)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)
        self.hop_num = hop_num
        self.feature_drop = nn.Dropout(fea_drop)
        self.layerRegular = []
        for i in range(self.hop_num):
            self.layerRegular.append(Parameter(torch.rand(1,)))
            self.register_parameter('layerRegular'+str(i),self.layerRegular[i])
    def edge_applying(self, edges):
        g = torch.ones(len(edges),1).squeeze()
        e = g * edges.dst['d'] * edges.src['d']
        return {'e': e, 'm': g}
    def forward(self, graph: DGLGraph,h):
        graph = graph.local_var()
        graph.ndata['h'] = h
        graph.apply_edges(self.edge_applying)
        feat = h
        degree_normal = graph.edata.pop('e')
        graph.ndata['h'] = feat
        graph.edata['e'] = degree_normal
        graph.update_all(fn.u_mul_e('h', 'e', '_'), fn.sum('_', 'h'))
        feat = graph.ndata.pop('h') * self.layerRegular[0] + (1 - self.layerRegular[0]) * feat
        feat_r = self.feature_drop(feat)
        for _ in range(1,self.hop_num):
            graph.ndata['h'] = feat
            graph.edata['e'] = degree_normal
            graph.update_all(fn.u_mul_e('h', 'e', '_'), fn.sum('_', 'h'))
            feat = graph.ndata.pop('h') * self.layerRegular[_] + (1 - self.layerRegular[_]) * feat
            feat = self.feature_drop(feat)
            feat_r = torch.cat([feat_r,feat],dim=1)
        return feat_r
class incepLayer(nn.Module):#incpeGCN
    def __init__(self,in_dim,hop_num,fea_drop):
        super(incepLayer, self).__init__()
        self.gate = nn.Linear(2 * in_dim, 1)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)
        self.hop_num = hop_num
        self.feature_drop = nn.Dropout(fea_drop)
        self.layerRegular = []
        for i in range(self.hop_num):
            param = []
            for j in range(i+1):
                param.append(Parameter(torch.rand(1,)))
                self.register_parameter('layerRegular'+str(i)+str(j),param[j])
            self.layerRegular.append(param)
    def edge_applying(self, edges):
        g = torch.ones(len(edges),1).squeeze()
        e = g * edges.dst['d'] * edges.src['d']
        return {'e': e, 'm': g}

    def forward(self, graph: DGLGraph, h):
        graph = graph.local_var()
        graph.ndata['h'] = h
        graph.apply_edges(self.edge_applying)
        feat_r = h
        degree_normal = graph.edata.pop('e')
        for _ in range(self.hop_num):
            feat = h
            for q in range(_+1):
                graph.ndata['h'] = feat
                graph.edata['e'] = degree_normal
                graph.update_all(fn.u_mul_e('h', 'e', '_'), fn.sum('_', 'h'))
                feat = graph.ndata.pop('h') * self.layerRegular[_][q] + (1 - self.layerRegular[_][q]) * feat
                # feat = feat *(1-self.alpha)+ self.alpha *h
                feat = self.feature_drop(feat)
            feat_r = torch.cat((feat_r,feat),dim=1)
        return feat_r
class PositionwiseFeedForward(nn.Module):#commom feature transformation
    def __init__(self, model_dim, d_hidden, dropout=0.5):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(model_dim, d_hidden)
        self.dropout = nn.Dropout(dropout)
        self.init()
    def forward(self, x):
        return self.dropout(F.relu(self.w_1(x)))
    def init(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.w_1.weight, gain=gain)
