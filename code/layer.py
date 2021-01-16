import torch
from torch import nn
from torch.nn import functional as F
import dgl.function as fn
from dgl.utils import expand_as_pair, check_eq_shape

# @article{wang2019dgl,
#     title={Deep Graph Library: A Graph-Centric, Highly-Performant Package for Graph Neural Networks},
#     author={Minjie Wang and Da Zheng and Zihao Ye and Quan Gan and Mufei Li and Xiang Song and Jinjing Zhou and Chao Ma and Lingfan Yu and Yu Gai and Tianjun Xiao and Tong He and George Karypis and Jinyang Li and Zheng Zhang},
#     year={2019},
#     journal={arXiv preprint arXiv:1909.01315}
# }
# 在网络中节点聚合的过程中，加入了权重
class SAGEConv(nn.Module):
    def __init__(self, in_feats, out_feats, 
                 aggregator_type,
                 feat_drop=0.5, activation=F.relu,
                 bias=True, norm=None):
        super(SAGEConv, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        # aggregator type: mean/gcn
        if aggregator_type != 'gcn':
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)

        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if self._aggre_type != 'gcn':
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)

        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def forward(self, graph, feat):
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = feat[0]
                feat_dst = feat[1]
            else:
                feat_src = feat_dst = feat
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]

            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.number_of_edges() == 0:
                graph.dstdata['neigh'] = torch.zeros(
                    feat_dst.shape[0], self._in_src_feats).to(feat_dst)
            
            elif self._aggre_type == 'mean':
                graph.srcdata['h'] = feat_src
                graph.update_all(fn.u_mul_e('h', 'weight', 'm'), fn.sum('m', 'neigh'))
                graph.update_all(fn.copy_e('weight', 'm'), fn.sum('m', 'ws'))
                h = graph.dstdata['neigh']
                ws = graph.dstdata['ws'].unsqueeze(1).clamp(min=1)
                h_neigh = h / ws
            elif self._aggre_type == 'gcn':
                check_eq_shape(feat)
                graph.srcdata['h'] = feat_src
                graph.dstdata['h'] = feat_dst    
                graph.update_all(fn.u_mul_e('h', 'weight', 'm'), fn.sum('m', 'neigh'))
                graph.update_all(fn.copy_e('weight', 'm'), fn.sum('m', 'ws'))
                h = graph.dstdata['neigh']
                ws = graph.dstdata['ws'].unsqueeze(1)
                h_neigh = (h + graph.dstdata['h']) / (ws + 1)
            else:
                raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))

            # GraphSAGE GCN does not require fc_self.
            if self._aggre_type == 'gcn':
                rst = self.fc_neigh(h_neigh)
            else:
                rst = (self.fc_self(h_self) + self.fc_neigh(h_neigh)) / 2

            return rst