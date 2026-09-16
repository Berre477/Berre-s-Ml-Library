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
    def __init__(self,data,parents = ()):
        self.data = np.array(data,dtype=np.float32)
        self.grad = np.zeros_like(self.data)
        self.prev_par = set(parents)
        self.vjp = None


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
            if node.vjp is not None:
                node.vjp()

    def __add__(self,other):
        other = other if isinstance(other,Tensor) else Tensor(other)
        out = Tensor(self.data + other.data,parents=(self,other))

        def vjp():
            self.grad += unbroadcast(out.grad,self.data.shape)
            other.grad += unbroadcast(out.grad,other.data.shape)

        out.vjp = vjp
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

    def __repr__(self):
        return f"Tensor has a shape of {self.data.shape}, and data = {self.data}"


    def __matmul__(self,other):
        other = other if isinstance(other,Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data,parents=(self,other))

        def vjp():
            self.grad += unbroadcast(out.grad @ other.data.T ,self.data.shape )
            other.grad +=unbroadcast(self.data.T @ out.grad,other.data.shape)

        out.vjp = vjp
        return out


    def __rmatmul__(self, other):
        return Tensor(other) @ self



    def sum(self,axis=None,keepdims=False):
        out = Tensor(self.data.sum(axis=axis,keepdims=keepdims),parents=(self,))

        def vjp():
            grad = out.grad
            if axis is not None and not keepdims:
                axes = [axis] if isinstance(axis,int) else sorted(axis)
                for ax in axes:
                    grad = np.expand_dims(grad,axis=ax)

            self.grad +=np.ones_like(self.data) * grad

        out.vjp = vjp
        return out
    
    def __pow__ (self,power):
        assert isinstance(power,(int,float)), "power must be an int or float"
        out = Tensor(self.data ** power,parents=(self,))

        def vjp():
            grad_val = (power *(self.data **(power-1))) * out.grad
            self.grad += unbroadcast(grad_val,self.data.shape)

        out.vjp = vjp

        return out

    def __truediv__(self,other):
        other = other if isinstance(other,Tensor) else Tensor(other)
        return self * (other** -1.0)


    def __rtruediv__(self, other):
        return Tensor(other) / self

    def relu(self):
        out = Tensor(np.maximum(0.0,self.data),parents=(self,))
        def vjp():
            self.grad +=(self.data > 0.0) *out.grad
        out.vjp = vjp
        return out


    def reshape(self,*shape):
        new_shape = shape[0] if isinstance(shape[0],(list,tuple)) else shape
        out = Tensor(self.data.reshape(new_shape),parents=(self,))
        def vjp():
            self.grad += out.grad.reshape(self.data.shape)
        out.vjp = vjp
        return out


    def T(self):
        out = Tensor(self.data.T,parents=(self,))
        def vjp():
            self.grad.T
        out.vjp = vjp
        return out

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)