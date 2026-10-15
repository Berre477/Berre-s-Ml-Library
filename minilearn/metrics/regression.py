import numpy as np


def mean_squared_error(ytrue,ypred):
    ytrue = np.asarray(ytrue,dtype=np.float32)
    ypred = np.asarray(ypred , dtype=np.float32)
    return float(np.mean((ytrue - ypred)** 2))

def root_mean_squared_error(ytrue,ypred):
    return float(np.sqrt(mean_squared_error(ytrue,ypred)))

def mean_absolute_error(ytrue,y_red):
    ytrue = np.asarray(ytrue,dtype=np.float32)
    ypred = np.asarray(ypred , dtype=np.float32)
    return float(np.mean(np.abs(ytrue - ypred)))


def r2_score(ytrue,ypred):
    ytrue = np.asarray(ytrue,dtype=np.float32)
    ypred = np.asarray(ypred,dtype=np.float32)
    total_ss = np.sum((ytrue - np.mean(ytrue))**2)
    if total_ss == 0.0:return 0.0
    res_ss = np.sum((ytrue - ypred)**2)
    return float(1.0 - (res_ss / total_ss))
    
