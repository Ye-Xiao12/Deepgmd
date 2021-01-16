import torch 
import numpy as np

__all__ = [
    'cal_ch',
]

# CHI : Calinski-Harabaz index
# BGSS / WGSS * (n-k) / (k - 1)
# BGSS (between-group sum of squared erro) 簇间分离度
# WGSS (withim-groups sum of squared error) 簇内紧密度
# Calinski T, Harabasz J. A dendrite method for cluster analysis[J]. Communications in Statistics-theory and Methods, 1974, 3(1): 1-27.
def cal_ch(community_matrix,distant):
    matrix = community_matrix
    if isinstance(community_matrix,(torch.Tensor)):
        matrix = community_matrix.to('cpu').detach().numpy()
    distant = distant ** 2
    d = 2 * np.sum(np.triu(distant,1)) / (matrix.shape[0] ** 2 - matrix.shape[0])
    cluster_num = 0
    wgss,bgss = 0,0
    for member in matrix.T:
        num = member.sum()
        cluster_num += 1
        if num <= 1:
            continue
        member = np.where(member == 1)[0]
        cluster_distant = distant[member][:,member]
        d_i = 2 * np.sum(np.triu(cluster_distant,1)) / (num ** 2 - num)
        wgss += (num - 1) * d_i
        bgss += (num - 1) * (d - d_i)
    bgss += (cluster_num - 1) * d
    return bgss / wgss * (distant.shape[0] - cluster_num) / (cluster_num - 1)