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
        self.prev_par = set(parents)
        self.backward = None


    def backward(self):
        """
        Executes reverse-mode diff using topological sort
        """

        topo = []
        vis= set()

        def build_topo(node):
            if node not in vis:
                vis.add(node)
                for parent in node.prev_par:
                    build_topo(parent)
                topo.append(node)

        build_topo(self)

        self.grad = np.ones_like(self.data) #(dOut/dOut = 1.0)

        #Travese DAG in reverse topological order
        for node in reversed(topo):
            node.backward()

    def __add__(self,other):
        other = other if isinstance(other,Tensor) else Tensor(other)
        out = Tensor(self.data + other.data,parents=(self,other))

        def vjp():
            self.grad += unbroadcast(out.grad,self.data.shape)
            other.grad += unbroadcast(out.grad,other.data.shape)

        out.backward = vjp
        return out


    def __sub__(self,other):
        return self + (-other)

    def __rsub__(self, other):
        return Tensor(other) - self

    def __radd__(self, other):
        return self + other


    def __neg__(self):
        return self * -1.0

    def __rmul__(self, other):
        return self * other

    def __mul__(self, other):
        other = other if isinstance(other,Tensor) else Tensor(other)
        out = Tensor(self.data * other.data,parents=(self,other))

        def vjp():
            self.grad += unbroadcast(other.data * out.grad,self.data.shape)
            other.grad += unbroadcast(self.data * out.grad ,other.data.shape)

        out.backward = vjp
        return out
    