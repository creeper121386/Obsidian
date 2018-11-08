#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import numpy.lib.user_array as UsrArry


def raise_err(func):
    def wrapper(self, other):
        if self.need_grad:
            raise RuntimeError(
                'Tensor in graph don\'t support in-place operation.')
        return func(self, other)
    return wrapper


def _get_grad(name, inputs):
    fnuc_dict = {
        'add': lambda x, y: (1, 1),
        'mul': lambda x, y: (y, x),
        'sub': lambda x, y: (1, -1),
        'neg': lambda x: -1,
        'div': lambda x, y: (1/y, -x/y**2),
        'pow': lambda x, y: (y*(x**(y-1)), np.log(x) * x**y),
        'abs': lambda x: 1,
        'matmul': lambda x, y: (),
        'transpose': lambda x, y: ()
    }
    return fnuc_dict[name](inputs)


class Tensor(UsrArry.container):
    def __init__(self, data, dtype=None, copy=True, need_grad=False,  bak_nodes=None, grad=0, node_name=None):
        '''
        Args:
        =====
        - data: must be a list or other iterable object.
        - bak_nodes: the computational node in the downstream.
        - grad: the grad of this This node.
        - need_grad: need grad or not.
        - cpt_func: the computation to get the current Tensor node.

        In a computational graph, a Tensor node looks like:

            Forward: Tensor.bak_nodes ->(cpt_func)-> Tensor
            Backward: Tensor.bak_nodes <-(grad_func)<- Tensor

        '''

        UsrArry.container.__init__(self, data, dtype=None, copy=True)
        # super(Tensor, self).__init__(data, dtype=None, copy=True)
        self.node_name = node_name
        self.cpt_name = None
        self.need_grad = need_grad
        self.grad = grad
        self.bak_nodes = bak_nodes

    def _add_to_graph(self, upper_nodes, name, other=None, self_in_right=False):
        '''
        Args:
        =====
        - upper nodes: upper node of current node in computational graph.
        - name: name of computation to `get` the current node.
        - other: node in the same level with current node (if exsit).
        - self_in_right: if current node is on the right of the opration, for example: 
            res = self - other, then `self_in_right` = False;
            res = other - self, then `self_in_right` = True;
        '''
        if isinstance(upper_nodes, Tensor):
            upper_nodes.need_grad = True
            upper_nodes.cpt_name = name
            if not other:
                upper_nodes.bak_nodes = (self, )
            elif self_in_right:
                upper_nodes.bak_nodes = (other, self)
            else:
                upper_nodes.bak_nodes = (self, other)

    def __str__(self):
        return super().__str__() + '\ndtype={}, size={}\n'.format(self.dtype, self.shape)

    # Reload oprations for Tensor:
    def __abs__(self):
        res = super().__abs__()
        if self.need_grad:
            self._add_to_graph(res, 'abs')
        return res

    def __neg__(self):
        res = super().__neg__()
        if self.need_grad:
            self._add_to_graph(res, 'neg')
        return res

    def __add__(self, other):
        res = super().__add__(other)
        if self.need_grad:
            self._add_to_graph(res, 'add', other)
        return res

    def __matmul__(self, other):
        res = Tensor(np.dot(self, other))
        if self.need_grad:
            self._add_to_graph(res, 'matmul', other)
        return res

    def __rmatmul__(self, other):
        res = Tensor(np.dot(other, self))
        if self.need_grad:
            self._add_to_graph(res, 'matmul', other, self_in_right=True)
        return res

    __radd__ = __add__

    @raise_err
    def __iadd__(self, other):
        return super().__iadd__(other)

    def __sub__(self, other):
        res = super().__sub__(other)
        if self.need_grad:
            self._add_to_graph(res, 'sub', other)
        return res

    def __rsub__(self, other):
        res = super().__rsub__(other)
        if self.need_grad:
            self._add_to_graph(res, 'sub', other, self_in_right=True)
        return res

    @raise_err
    def __isub__(self, other):
        return super().__isub__(other)

    def __mul__(self, other):
        res = super().__mul__(other)
        if self.need_grad:
            self._add_to_graph(res, 'mul', other)
        return res

    __rmul__ = __mul__

    @raise_err
    def __imul__(self, other):
        return super().__imul__(other)

    def __truediv__(self, other):
        res = Tensor(np.array(self) / np.array(other))
        if self.need_grad:
            self._add_to_graph(res, 'div', other)
        return res

    def __rtruediv__(self, other):
        res = Tensor(np.array(other) / np.array(self))
        if self.need_grad:
            self._add_to_graph(res, 'div', other, self_in_right=True)
        return res

    @raise_err
    def __itruediv__(self, other):
        return Tensor(np.array(self) / np.array(other))

    @raise_err
    def __idiv__(self, other):
        return super().__idiv__(other)

    def __mod__(self, other):
        res = super().__mod__(other)
        if self.need_grad:
            self._add_to_graph(res, 'mod', other)
        return res

    def __rmod__(self, other):
        res = super().__rmod__(other)
        if self.need_grad:
            self._add_to_graph(res, 'mod', other, self_in_right=True)
        return res

    @raise_err
    def __imod__(self, other):
        return super().__imod__(other)

    def __divmod__(self, other):
        print('divmoding')
        res = super().__divmod__(other)
        if self.need_grad:
            self._add_to_graph(res, 'divmod', other)
        return res

    def __rdivmod__(self, other):
        res = super().__rdivmod__(other)
        if self.need_grad:
            self._add_to_graph(res, 'divmod', other, self_in_right=True)
        return res

    def __pow__(self, other):
        res = super().__pow__(other)
        if self.need_grad:
            self._add_to_graph(res, 'pow', other)
        return res

    def __rpow__(self, other):
        res = super().__rpow__(other)
        if self.need_grad:
            self._add_to_graph(res, 'pow', other, self_in_right=True)
        return res

    @raise_err
    def __ipow__(self, other):
        return super().__ipow__(other)

    def backward(self):
        self.grad = 1
        now = self
        baks = now.bak_nodes
        while len(baks):
            bak_grads = now.grad * _get_grad(now.cpt_name, baks)
            for i in range(len(baks)):
                baks[i].grad += bak_grads[i]
                pass
            
                



    # matrix computation method:
    def transpose(self):
        res = Tensor(np.transpose(self))
        if self.need_grad:
            self._add_to_graph(res, 'transpose')
        return res

    T = transpose

    dot = __matmul__


if __name__ == '__main__':
    t1 = Tensor([1, 2, 3, 4])
    t2 = Tensor([4, 3, 2, 1])
    print(type(t1.T))
