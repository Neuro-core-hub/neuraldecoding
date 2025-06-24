import numpy as np

def calc_corr(y1,y2):
    """Calculates the correlation between y1 and y2 (tensors)"""
    corr = []
    for i in range(y1.shape[1]):
        corr.append(np.corrcoef(y1[:, i], y2[:, i])[1, 0])
    return corr
