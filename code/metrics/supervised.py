import warnings
import torch
import numpy as np
import pandas as pd 

__all__ = [
    'cal_nmi',
    'cal_overlap',
]

# 归一化互信息（NMI)
# overlap
def cal_nmi(X,Y):
    """Compute NMI between two overlapping community covers.

    Parameters
    ----------
    X : array-like, shape [N, m]
        Matrix with samples stored as columns.
    Y : array-like, shape [N, n]
        Matrix with samples stored as columns.

    Returns
    -------
    nmi : float
        Float in [0, 1] quantifying the agreement between the two partitions.
        Higher is better.

    References
    ----------
    McDaid, Aaron F., Derek Greene, and Neil Hurley.
    "Normalized mutual information to evaluate overlapping
    community finding algorithms."
    arXiv preprint arXiv:1110.2515 (2011).

    """
    if not ((X == 0) | (X == 1)).all():
        raise ValueError("Observed module matrix should be a binarray matrix")
    if not ((Y == 0) | (Y == 1)).all():
        raise ValueError("Known module matrix should be a binarray matrix")

    if X.shape[1] > X.shape[0]:
        warnings.warn("It seems that you forget to transpose thr F matrix")
        X = X.T 
    if Y.shape[1] > Y.shape[0]:
        warnings.warn("It seems that you forget to transpose thr F matrix")
        Y = Y.T
    X,Y = X.T , Y.T 
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Dimension don't match")
    
    h = lambda w,n : 0 if w == 0 else -w * np.log2(w / n)
    
    def H(x,y):
        """Compute conditional entropy between two vectors."""
        a = (1 - x).dot(1 - y)
        d = x.dot(y)
        c = (1 - y).dot(x)
        b = (1 - x).dot(y)
        n = len(x)

        if h(a, n) + h(d, n) >= h(b, n) + h(c, n):
            return h(a, n) + h(b, n) + h(c, n) + h(d, n) - h(b + d, n) - h(a + c, n)
        else:
            return h(c + d, n) + h(a + b, n)
    
    def H_uncond(X):
        """Compute unconditional entropy of a single binary matrix."""
        return sum(h(x.sum(), len(x)) + h(len(x) - x.sum(), len(x)) for x in X)

    def H_cond(X, Y):
        """Compute conditional entropy between two binary matrices."""
        m, n = X.shape[0], Y.shape[0]
        scores = np.zeros([m, n])
        for i in range(m):
            for j in range(n):
                scores[i, j] = H(X[i], Y[j])
        return scores.min(axis=1).sum()
    
    H_X = H_uncond(X)
    H_Y = H_uncond(Y)
    I_XY = 0.5 * (H_X + H_Y - H_cond(X, Y) - H_cond(Y, X))
    return I_XY / max(H_X, H_Y)

# calculate recovery and relance
def cal_overlap(X,Y):
    if not ((X == 0) | (X == 1)).all():
        raise ValueError("Observed module matrix should be a binarray matrix")
    if not ((Y == 0) | (Y == 1)).all():
        raise ValueError("Known module matrix should be a binarray matrix")

    if X.shape[1] > X.shape[0]:
        warnings.warn("It seems that you forget to transpose thr F matrix")
        X = X.T 
    if Y.shape[1] > Y.shape[0]:
        warnings.warn("It seems that you forget to transpose thr F matrix")
        Y = Y.T
    X,Y = X.T , Y.T 
    if X.shape[1] != Y.shape[1]:
        raise ValueError("Dimension don't match")

    m,n = X.shape[0],Y.shape[0]
    overlap = np.zeros((m,n))
    length = X.shape[1]
    for i in range(m):
        for j in range(n):
            a = X[i].dot(Y[j])
            b = (1 - X[i]).dot(1 - Y[j])
            overlap[i][j] = a / (length - b)
    
    relevance = np.array([max(elem) for elem in overlap]).mean()
    recovery =  np.array([max(elem) for elem in overlap.T]).mean()
    return relevance,recovery