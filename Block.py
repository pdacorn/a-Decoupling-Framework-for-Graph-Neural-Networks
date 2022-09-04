import torch
from torch import nn
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.function as fn
from torch.nn.parameter import Parameter
from .layer import *
class JknetBlcok(nn.Module):#JKNET
    def __init__(self,
                 in_feats: int,
                 infeat:int,
                 hidden_dim: int,
                 hop_num,
                 feat_drop,
                 feed_forward=True,
                 ):
        super(JknetBlcok, self).__init__()
        self._in_feats = in_feats
        self.infeat = infeat
        self._out_feats = hidden_dim
        self.hop_num = hop_num
        self.feed_forward = feed_forward
        self.feat_drop = feat_drop
        self.prop = JKnetLayer(in_dim=self._in_feats,hop_num=self.hop_num,fea_drop=self.feat_drop)
        self.feed_forward = PositionwiseFeedForward(model_dim=self.infeat, d_hidden=self._out_feats)  # entity feed forward
        self.dropF = nn.Dropout(self.feat_drop)
    def forward(self, graph: DGLGraph, features):
        graph = graph.local_var()
        h = features
        rst = self.prop(graph, h)#propagation
        r = rst
        if not self.feed_forward:
            return F.elu(rst)
        rst_ff = self.feed_forward(rst)
        # rst is passed to the adaptive receptive field, and r is passed to the next BLOCK
        return rst_ff,r

class incepBlcok(nn.Module):#INCEPGCN
    def __init__(self,
                 in_feats: int,
                 infeat:int,
                 hidden_dim: int,
                 hop_num,
                 feat_drop,
                 feed_forward=True,
                 ):
        super(incepBlcok, self).__init__()
        self._in_feats = in_feats
        self.infeat = infeat
        self._out_feats = hidden_dim
        self.hop_num = hop_num
        self.feed_forward = feed_forward
        self.feat_drop = feat_drop
        self.prop = incepLayer(in_dim=self._in_feats,hop_num=self.hop_num,fea_drop=self.feat_drop)
        self.feed_forward = PositionwiseFeedForward(model_dim=self.infeat, d_hidden=self._out_feats)  # entity feed forward
    def forward(self, graph: DGLGraph, features):
        graph = graph.local_var()
        h = features
        rst = self.prop(graph, h)#propagation
        r = rst
        if not self.feed_forward:
            return F.elu(rst)

        rst_ff = self.feed_forward(rst)

        # rst is passed to the adaptive receptive field, and r is passed to the next BLOCK
        return rst_ff,r
class GCNBlock(nn.Module):#GCN
    def __init__(self,
                 in_feats: int,
                 hidden_dim: int,
                 hop_num,
                 feat_drop,
                 feed_forward=True,
                 ):
        super(GCNBlock, self).__init__()
        self._in_feats = in_feats
        self._out_feats = hidden_dim
        self.hop_num = hop_num
        self.feed_forward = feed_forward
        self.feat_drop = feat_drop
        self.prop = GCNLayer(in_dim=self._in_feats,hop_num=self.hop_num,fea_drop=self.feat_drop)
        self.feed_forward = PositionwiseFeedForward(model_dim=self._in_feats, d_hidden=self._in_feats)  # entity feed forward
    def forward(self, graph: DGLGraph, features):
        graph = graph.local_var()
        h = features
        rst = self.prop(graph, h)#propagation
        r = rst
        resval = features# residual
        if not self.feed_forward:
            return F.elu(rst)

        rst_ff = self.feed_forward(rst)
        rst = rst_ff + resval
        # rst is passed to the adaptive receptive field, and r is passed to the next BLOCK
        return rst,r
class APPNPBlock(nn.Module):#APPNP
    def __init__(self,
                 in_feats: int,
                 hidden_dim: int,
                 alpha,
                 hop_num,
                 feat_drop,
                 feed_forward=True,
                 ):
        super(APPNPBlock, self).__init__()
        self._in_feats = in_feats
        self._out_feats = hidden_dim
        self.alpha = alpha
        self.hop_num = hop_num
        self.feed_forward = feed_forward
        self.feat_drop = feat_drop
        self.prop = APPNPLayer(in_dim=self._in_feats,hop_num=self.hop_num,fea_drop=self.feat_drop,alpha=self.alpha)
        self.feed_forward = PositionwiseFeedForward(model_dim=self._in_feats, d_hidden=self._in_feats)  # entity feed forward
    def forward(self, graph: DGLGraph, features):
        graph = graph.local_var()
        h = features
        rst = self.prop(graph, h)#propagation
        r = rst
        resval = features# residual
        if not self.feed_forward:
            return F.elu(rst)
        rst_ff = self.feed_forward(rst)
        rst = rst_ff + resval
        # rst is passed to the adaptive receptive field, and r is passed to the next BLOCK
        return rst,r

