import numpy as np


def unbroadcast(grad,target_shape):
    """ 
    Sums out broadcasted dimensions so gradient shape matches parameter shape
    """
    #Sum extra dimennions added
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)

    #sum over dimensions where the target size is 1

    for i in range(len(target_shape)):
        if target_shape[i] == 1 :
            grad = grad.sum(axis = i,keepdims = True)

    return grad

class Tensor:
    def __init__(self,data,parents):
        self.data = np.array(data,dtype=np.float32)
        self.grad = np.zeros_like(self.data)
