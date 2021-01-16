import dgl
import torch
import numpy as np
import pandas as pd 
from sklearn.cluster import KMeans
from metrics import *
import scipy

data_name = ["ecoli_colombos", "ecoli_dream5", "yeast_gpl2529", 
"yeast_dream5", "synth_ecoli", "synth_yeast"]

# 加载数据
def load_data(path):
    print('loading '+path.split('/')[-2]+'...')
    feature = np.load(path+'feat.npy')
    g = dgl.DGLGraph()
    src,dst,weight= [],[],[]
    with open(path+'coexp_net.txt','r') as file:
        for line in file:
            a,b,w = int(line.split()[0]),int(line.split()[1]),float(line.split()[2])
            src.append(a)
            dst.append(b)
            weight.append(w)
    g.add_nodes(feature.shape[0])         # 添加节点
    g.add_edges(src+dst,dst+src)          # 添加双向边
    dgl.add_self_loop(g) 
    g.ndata['feat'] = torch.FloatTensor(feature)
    g.edata['weight'] = torch.FloatTensor(weight+weight)

    id_gene,gene_id = {},{}
    with open(path+'id_gene.txt','r') as file:
        for line in file:
            num,gene = int(line.split()[0]),line.split()[1]
            id_gene[num] = gene
            gene_id[gene] = num 
    return id_gene,gene_id,g

# 提前结束，保存模型
class EarlyStopping(object):
    def __init__(self, args):
        self.filename = args.path+'model.pth'
        self.patience = args.patience
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, model):
        if self.best_loss is None:
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss):
            self.counter += 1
            # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))

# transform affiliation matrix into N*m
def cal_matrix(minimal_path,gene):
    minimal = pd.read_json(minimal_path)
    tmp = minimal.values.reshape(minimal.shape[0]*minimal.shape[1],)
    tmp = np.array(list(set(tmp)))
    gene_mark = [elem in tmp for elem in gene]

    data = minimal.T.apply(pd.value_counts)
    data = pd.DataFrame(data,index = gene)
    matrix = np.array(data)
    matrix[np.isnan(matrix)] = 0
    return matrix[gene_mark],gene_mark 

# kmean
def kmean(feat,n_clusters = 100):
    knn = KMeans(n_clusters = n_clusters,init = 'k-means++',n_init = 10,max_iter = 300,random_state=0)
    knn.fit(feat)
    matrix = np.zeros((feat.shape[0],n_clusters))
    for i in range(feat.shape[0]):
        matrix[i][knn.labels_[i]] = 1
    return matrix

# hcluster
def hcluster(feat, n_cluster=100,metric='euclidean',method='ward'):
    import scipy.cluster.hierarchy as hcl

    disMat = hcl.distance.pdist(X = feat, metric=metric)
    linkage = hcl.linkage(disMat, method='ward')
    # hcluster.dendrogram(linkage,  leaf_font_size=10.)
    labels = hcl.fcluster(linkage, n_cluster, criterion='maxclust')
    matrix = np.zeros((feat.shape[0],n_cluster))
    for i in range(labels.shape[0]):
        matrix[i][labels[i]-1] = 1
    return matrix

# 计算相似度
def simdist(emb, simdist_function="pearson_correlation_abs", similarity=True, **kwargs):
    choices = {
        "pearson_correlation": lambda emb: np.corrcoef(emb),
        "pearson_correlation_abs": lambda emb: np.abs(np.corrcoef(emb))
    }

    func = choices[simdist_function]
    simdist_matrix = func(emb)

    if similarity == True:
        pass
    else:
        simdist_matrix =  (-simdist_matrix) + simdist_matrix.max().max()
        print(simdist_matrix)
    return simdist_matrix
    
# adjust matrix
def adjust(matrix,mark,top_k = 3):
    tmp = matrix[mark]
    mark_2 = np.where(tmp.sum(0) >= top_k)[0]
    tmp = tmp[:,mark_2]
    return tmp

# 计算F1score 和 Nmi 指标
def cal_metric(args):
    gene_id = {}
    with open(args.path+'id_gene.txt','r') as file:
        for line in file:
            num,gene = int(line.split()[0]),line.split()[1]
            gene_id[gene] = num 

    data = args.path.split('/')[-2]
    print(type(data_name[0]))
    minimal_path,strict_path = args.path+'knownmodules/minimal.json',args.path+'knownmodules/strict.json'
    minimal_matrix,minimal_mark = cal_matrix(minimal_path,gene_id.keys())
    strict_matrix,strict_mark = cal_matrix(strict_path,gene_id.keys())
    
    emb = np.load(args.path+'emb.npy')

    # 数据标准化
    eps = 1e-5
    mu = np.mean(emb,axis = 0)
    sigma = np.std(emb,axis = 0)
    tmp = (emb - mu) / (sigma + eps)

    print('deepgmd_cluster:')
    matrix = hcluster(tmp,args.k)
    res_minimal = adjust(matrix,minimal_mark,top_k = 3)
    res_strict = adjust(matrix,strict_mark,top_k = 3)

    relevance,recovery = cal_overlap(res_minimal,minimal_matrix)
    nmi = cal_nmi(res_minimal,minimal_matrix)
    print("minimal:  F1 score:{:.4f} nmi:{:.4f}".format((relevance+recovery)/2,nmi))
    relevance,recovery = cal_overlap(res_strict,strict_matrix)
    nmi = cal_nmi(res_strict,strict_matrix)
    print("strict:   F1 score:{:.4f} nmi:{:.4f}".format((relevance+recovery)/2,nmi))

    print('deepgmd:')
    matrix = emb
    matrix[matrix > args.threshold] = 1
    matrix[matrix < args.threshold] = 0
    res_minimal = adjust(matrix,minimal_mark,top_k = 3)
    res_strict = adjust(matrix,strict_mark,top_k = 3)

    relevance,recovery = cal_overlap(res_minimal,minimal_matrix)
    nmi = cal_nmi(res_minimal,minimal_matrix)
    print("minimal:  F1 score:{:.4f} nmi:{:.4f}".format((relevance+recovery)/2,nmi))
    relevance,recovery = cal_overlap(res_strict,strict_matrix)
    nmi = cal_nmi(res_strict,strict_matrix)
    print("strict:   F1 score:{:.4f} nmi:{:.4f}".format((relevance+recovery)/2,nmi))