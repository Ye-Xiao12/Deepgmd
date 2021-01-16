import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import argparse,json,time
import tqdm
import numpy as np 
import torch.nn.init as init 
from layer import SAGEConv
from torch.nn.functional import normalize
from util import load_data,cal_metric,EarlyStopping
    
# 生成输入向量
def load_subtensor(g, input_nodes, device):
    batch_inputs = g.ndata['feat'][input_nodes].to(device)
    return batch_inputs

# 负边采样: 按照节点度数采样
class NegativeSampler(object):
    def __init__(self, g, k, neg_share=False):
        self.weights = g.in_degrees().float() ** 0.75
        self.k = k
        self.neg_share = neg_share

    def __call__(self, g, eids):
        src, _ = g.find_edges(eids)
        n = len(src)
        if self.neg_share and n % self.k == 0:
            dst = self.weights.multinomial(n, replacement=True)
            dst = dst.view(-1, 1, self.k).expand(-1, self.k, -1).flatten()
        else:
            dst = self.weights.multinomial(n*self.k, replacement=True)
        src = src.repeat_interleave(self.k)
        return src, dst

# Sigmoid Loss: Sigmoid Loss 损失函数
class SigmoidLoss(nn.Module):
    def __init__(self, g, balance = True):
        super(SigmoidLoss, self).__init__()
        self.balance = balance
        self.prob = (g.num_edges() / (g.num_nodes()**2 - g.num_nodes())) * 2    # 图中任意两点有边概率
        self.eps = -np.log(1 - self.prob)

    def forward(self, block_outputs, pos_graph, neg_graph):
        with pos_graph.local_scope():
            pos_graph.ndata['h'] = block_outputs
            pos_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            pos_score = pos_graph.edata['score']
        with neg_graph.local_scope():
            neg_graph.ndata['h'] = block_outputs
            neg_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            neg_score = neg_graph.edata['score']

        ratio = neg_graph.num_edges() / pos_graph.num_edges()
        # sigmoid_loss for loss of edge
        loss_edges = -torch.mean(torch.log(torch.pow(1 + torch.exp(-pos_score-self.eps),-1)))
        # sigmoid_loss for loss of nonedge
        loss_nonedges = torch.mean(neg_score)
        if self.balance == True:
            loss = loss_edges + loss_nonedges
        else:
            loss = loss_edges + loss_nonedges * ratio

        return loss
           
# Berpo Loss: Berpo损失函数
class BerpoLoss(nn.Module):
    def __init__(self,g, balance=True):
        super(BerpoLoss, self).__init__()
        self.balance = balance
        self.prob = (g.num_edges() / (g.num_nodes()**2 - g.num_nodes())) * 2    # 图中任意两点有边概率
        self.eps = -np.log(1 - self.prob)

    def forward(self, block_outputs, pos_graph, neg_graph):
        with pos_graph.local_scope():
            pos_graph.ndata['h'] = block_outputs
            pos_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            pos_score = pos_graph.edata['score']
        with neg_graph.local_scope():
            neg_graph.ndata['h'] = block_outputs
            neg_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            neg_score = neg_graph.edata['score']

        ratio = neg_graph.num_edges() / pos_graph.num_edges()
        #Bernoulli-Possion loss for edge
        edge_dots = torch.sum(pos_score, dim = 1)
        loss_edges = -torch.mean(torch.log(-torch.expm1(-self.eps - edge_dots)))
        #Bernoulli-Possion loss for nonedge
        loss_nonedges = torch.mean(neg_score)
        if self.balance == True:
            loss = loss_edges + loss_nonedges
        else:
            loss = loss_edges + loss_nonedges * ratio

        return loss

# Deepgmd
class Deepgmd(nn.Module):
    def __init__(self, hidden_dim, n_input, n_z, 
                activation=F.relu,dropout = 0,
                aggregator_type = 'gcn', 
                batch_norm = True):
        super(Deepgmd, self).__init__()

        self.batch_norm = batch_norm
        self.aggregator_type = aggregator_type
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.hidden_dim = hidden_dim
        self.batch_norm = batch_norm
        self.n_z = n_z
        
        # GNN for inter information
        self.gcn_layers = nn.ModuleList()

        self.gcn_layers.append(
            SAGEConv( in_feats=n_input, out_feats=hidden_dim[0],
                aggregator_type = self.aggregator_type, bias=False))
        for idx in range(len(hidden_dim) - 1):
            self.gcn_layers.append(
                SAGEConv( in_feats=hidden_dim[idx], out_feats=hidden_dim[idx + 1],
                    aggregator_type = self.aggregator_type, bias=False))
        self.gcn_layers.append(
            SAGEConv( in_feats=hidden_dim[-1], out_feats=n_z,
                aggregator_type = self.aggregator_type, bias=False))

        # batch_norm 层
        if batch_norm:
            self.batch_norm = [
                torch.nn.BatchNorm1d(dim,affine=False, track_running_stats=False) 
                for dim in hidden_dim
            ]
        else:
            self.batch_norm = None
    
    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.gcn_layers, blocks)):
            h = layer(block, h)
            if l != len(self.gcn_layers) - 1:
                h = self.activation(h)
                if self.batch_norm is not None:
                    x = self.batch_norm[l](x)
                h = self.dropout(h)
        return F.relu(h)
    
    def inference(self, g, x, batch_size, device):
        nodes = torch.arange(g.number_of_nodes())

        for l, layer in enumerate(self.gcn_layers):
            y = torch.zeros(g.number_of_nodes(), self.hidden_dim[l] if l != len(self.gcn_layers) - 1 else self.n_z)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,torch.arange(g.number_of_nodes()),sampler,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=args.num_workers)

            for step,(input_nodes, output_nodes, blocks) in enumerate(dataloader):
                block = blocks[0].to(device)

                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l != len(self.gcn_layers) - 1:
                    h = self.activation(h)
                    if self.batch_norm is not None:
                        x = self.batch_norm[l](x)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
        return F.relu(y)

def run(args,device):
    # 设置随机参数
    torch.manual_seed(args.seed)            # 为CPU设置随机种子
    torch.cuda.manual_seed(args.seed)       # 为GPU设置随机种子
    np.random.seed(args.seed)               # 为numpy设置随机种子

    # 加载数据
    _,_,g = load_data(args.path)
    n_edges = g.num_edges()
    train_seeds = torch.tensor(np.arange(g.num_edges()))
    train_nids = g.nodes()

    sampler = dgl.dataloading.MultiLayerNeighborSampler([int(fanout) for fanout in args.fan_out.split(',')])
    NegativeSampler_Uniform = dgl.dataloading.negative_sampler.Uniform(args.num_negs)       # 平均负采样
    dataloader = dgl.dataloading.EdgeDataLoader(
        g, train_seeds, sampler, exclude='reverse_id',
        # For each edge with ID e in Reddit dataset, the reverse edge is e ± |E|/2.
        reverse_eids=torch.cat([
            torch.arange(n_edges // 2, n_edges),
            torch.arange(0, n_edges // 2)]),
        negative_sampler = NegativeSampler(g, args.num_negs) if args.Negative_sample == 'deg' 
                else NegativeSampler_Uniform,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        pin_memory=True,
        num_workers=args.num_workers)

    model = Deepgmd(hidden_dim=args.hidden_dim, 
            n_input=args.n_input, n_z=args.n_z, 
            aggregator_type = args.aggregator_type, 
            activation=F.relu, dropout = args.dropout)
    print(model)
    model = model.to(device)
    
    GnnLoss_fcn = SigmoidLoss(g)    # Sigmoid交叉熵损失
    GnnLoss_fcn = GnnLoss_fcn.to(device)

    stopper = EarlyStopping(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size = 10,gamma = 0.9)

    begin = time.time()
    model.train()
    for epoch in range(args.num_epochs):
        loss_arr = []
        for step, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(dataloader):
            gnn_inputs = load_subtensor(g, input_nodes, device)
            pos_graph = pos_graph.to(device)
            neg_graph = neg_graph.to(device)
            blocks = [block.int().to(device) for block in blocks]

            gnn_pred = model(blocks, gnn_inputs)

            gnn_loss = GnnLoss_fcn(gnn_pred, pos_graph, neg_graph)
            loss = gnn_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss_arr.append(loss.detach().cpu().numpy())
            med_loss = np.mean(loss_arr[-args.log_every:])
            if step % args.log_every == 0:
                gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
                print('Epoch {:05d} | Step {:05d} | Loss {:.4f} | Time: {:.4f} | GPU {:.1f} MB'.format(
                        epoch, step, med_loss.item(), time.time()-begin,gpu_mem_alloc))
        
        scheduler.step()
        loss_epoch = np.mean(loss_arr)
        # early_stop = stopper.step(loss_epoch, model)
        # if early_stop:
        #     break
    
    # model.load_state_dict(torch.load(stopper.filename))
    model.eval()
    gnn_emb = model.inference(g, g.ndata['feat'], args.batch_size, device)
    gnn_emb = gnn_emb.detach().cpu().numpy()
    np.save(args.path+'emb.npy',gnn_emb)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = ["ecoli_colombos", "ecoli_dream5", "yeast_gpl2529", "yeast_dream5", 
            "synth_ecoli", "synth_yeast"]
    data_name = data[0]
    print('data:',data_name)

    # 参数设置
    argparser = argparse.ArgumentParser("Graph-Sage")
    argparser.add_argument('--path', type=str, default='../data/ecoli_colombos/')
    argparser.add_argument('--Negative-sample', type=str, default='deg')
    argparser.add_argument('--aggregator-type', type=str, default='gcn')
    argparser.add_argument('--lr', type=float, default=1e-3)
    argparser.add_argument('--seed', type=int, default=0)
    argparser.add_argument('--weight-decay', type=float, default=1e-4)
    argparser.add_argument('--num-epochs', type=int, default=200)
    argparser.add_argument('--patience', type=int, default=10)
    argparser.add_argument('--batch-size',type=int, default=200)
    argparser.add_argument('--log-every', type=int, default=50)
    argparser.add_argument('--num-workers', type=int, default=4)
    argparser.add_argument('--hidden-dim', type=list, default=[700])
    argparser.add_argument('--fan-out', type=str, default='100,100')
    argparser.add_argument('--dropout', type=float, default=0.5) 
    argparser.add_argument('--threshold', type=float, default=0.5) 
    argparser.add_argument('--num-negs', type=int, default=10)
    argparser.add_argument('--n-z', type=int, default=128)
    argparser.add_argument('--k', type=int, default=100)
    argparser.add_argument('--n-input', type=int, default=500)
    args = argparser.parse_args()

    # nodes: 2522    
    # edges: 16687
    if data_name == 'ecoli_colombos':
        args.path = '../data/ecoli_colombos/'
        args.Negative_sample = 'uniform'
        args.aggregator_type = 'mean'
        args.hidden_dim = [500]
        args.batch_size = 500
        args.log_every = 500
        args.num_negs = 5
        args.k = 100
        args.n_z = 150

    # nodes: 2458
    # edges: 15928
    if data_name == 'ecoli_dream5':
        args.path = '../data/ecoli_dream5/'
        args.Negative_sample = 'uniform'
        args.aggregator_type = 'mean'
        args.hidden_dim=[500]
        args.batch_size=500
        args.log_every = 50
        args.num_negs = 10
        args.k = 100
        args.n_z = 128

    # nodes: 3178
    # edges: 26865
    if data_name == 'yeast_gpl2529':
        args.path = '../data/yeast_gpl2529/'
        args.Negative_sample = 'unifrom'
        args.aggregator_type = 'mean'
        args.threshold = 0.5
        args.hidden_dim = [1000]
        args.batch_size = 500
        args.log_every = 50
        args.num_negs = 40
        args.k = 50
        args.n_z = 128

    # nodes: 3292
    # edges: 28338
    if data_name == 'yeast_dream5':
        args.path = '../data/yeast_dream5/'
        args.Negative_sample = 'uniform'
        args.aggregator_type = 'gcn'
        args.threshold = 0.5
        args.hidden_dim = [500]
        args.batch_size = 500
        args.log_every = 100
        args.num_negs = 10
        args.k = 50
        args.n_z = 110
    
    # nodes: 1509
    # edges: 45529
    if data_name == 'synth_ecoli':
        args.path = '../data/synth_ecoli/'
        args.Negative_sample = 'uniform'
        args.aggregator_type = 'gcn'
        args.hidden_dim = [500]
        args.batch_size = 1000
        args.log_every = 50
        args.num_negs = 10
        args.k = 100
        args.n_z = 128
    
    # nodes: 1790
    # edges: 64097
    if data_name == 'synth_yeast':
        args.path = '../data/synth_yeast/'
        args.Negative_sample = 'uniform'
        args.aggregator_type = 'gcn'
        args.threshold = 0.5
        args.hidden_dim = [500]
        args.batch_size = 1000
        args.log_every = 50
        args.num_negs = 10
        args.k = 60
        args.n_z = 140
    
    run(args,device)
    cal_metric(args)