#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import numpy.lib.user_array as UsrArry

class Tensor(UsrArry.container):
    def __init__(self, data, dtype=None, copy=True, in_graph=False, upper_nodes=None, bak_nodes=None, grad=None):
        '''
        paramsters:

        - data: must be a list or other iterable object.
        - upper_node: the upper computational node of this Tensor.
        - bak_node: the computational node in the downstream.
        - grad: the grad of this This node.
        - need_grad: need grad or not.
        - cpt_func: the computation to get the current Tensor node.

        In a computational graph, a Tensor node looks like:
        
            Forward: Tensor.bak_nodes ->(Tensor.cpt_func)-> Tensor --> Tensor.upper_nodes
            Backward: Tensor.bak_nodes <-(Tensor.grad_func)<- Tensors
        '''

        UsrArry.container.__init__(self, data, dtype=None, copy=True)
        # super(Tensor, self).__init__(data, dtype=None, copy=True)
        # self.data = np.array(data)

        self.in_graph = in_graph
        # self.cpt_func = get_cpt_func(name)
        # self.grad_func = get_cpt_func(name, grad=True)
        self.grad = np.ones(self.shape) if self.in_graph else None
        self.upper_nodes = upper_nodes
        self.bak_nodes = bak_nodes

    def _add_to_graph(self):
        pass

    def __str__(self):
        return super().__str__() + ', dtype={}, size={}\n'.format(self.dtype, self.shape)

    def __abs__(self):
        return super().__abs__()

    def __neg__(self):
        return super().__neg__()

    def __add__(self, other):
        return super().__add__(other)

    __radd__ = __add__

    def __iadd__(self, other):
        return super().__iadd__(other)

    def __sub__(self, other):
        return super().__sub__(other)

    def __rsub__(self, other):
        return super().__rsub__(other)

    def __isub__(self, other):
        return super().__isub__(other)

    def __mul__(self, other):
        return super().__mul__(other)

    __rmul__ = __mul__

    def __imul__(self, other):
        return super().__imul__(other)

    def __div__(self, other):
        return super().__div__(other)

    def __rdiv__(self, other):
        return super().__rdiv__(other)

    def __idiv__(self, other):
        return super().__idiv__(other)





    def backward(self):
        reader = self.bak_nodes
        while 1: 
            for x in reader.bak_nodes:
                
                reader = reader.bak_nodes
        
    


if __name__ == '__main__':
    t1 = Tensor([1, 2, 3, 4])
    print(t1)
