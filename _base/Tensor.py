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


class Tensor(UsrArry.container):
    def __init__(self, data, dtype=None, copy=True, need_grad=False, upper_nodes=None, bak_nodes=None, grad=None):
        '''
        paramsters:

        - data: must be a list or other iterable object.
        - upper_nodes: the upper computational node of this Tensor.
        - bak_nodes: the computational node in the downstream.
        - grad: the grad of this This node.
        - need_grad: need grad or not.
        - cpt_func: the computation to get the current Tensor node.

        In a computational graph, a Tensor node looks like:

            Forward: Tensor.bak_nodes ->(cpt_func)-> Tensor --> Tensor.upper_nodes
            Backward: Tensor.bak_nodes <-(grad_func)<- Tensor
        '''

        UsrArry.container.__init__(self, data, dtype=None, copy=True)
        # super(Tensor, self).__init__(data, dtype=None, copy=True)

        self.cpt_name = None
        self.need_grad = need_grad
        self.grad = grad
        # self.upper_nodes = upper_nodes
        self.bak_nodes = bak_nodes

    def _add_to_graph(self, upper_nodes, name, other=None):
        if not isinstance(upper_nodes, Tensor):
            raise TypeError(
                'Input must be a Tensor, but got type {}'.format(type(upper_nodes)))
        # self.upper_nodes = upper_nodes

        upper_nodes.need_grad = True
        upper_nodes.cpt_name = name
        upper_nodes.bak_nodes = (self, ) if not other else (self, other)

        # if isinstance(other, Tensor) and other.need_grad:
        #     other.upper_nodes = upper_nodes

    def __str__(self):
        return super().__str__() + ', dtype={}, size={}\n'.format(self.dtype, self.shape)

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
            self._add_to_graph(res, 'sub', other)
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

    def __div__(self, other):
        res = super().__div__(other)
        if self.need_grad:
            self._add_to_graph(res, 'div', other)
        return res

    def __rdiv__(self, other):
        res = super().__rdiv__(other)
        if self.need_grad:
            self._add_to_graph(res, 'div', other)
        return res

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
            self._add_to_graph(res, 'mod', other)
        return res

    @raise_err
    def __imod__(self, other):
        return super().__imod__(other)

    def __divmod__(self, other):
        res = super().__divmod__(other)
        if self.need_grad:
            self._add_to_graph(res, 'divmod', other)
        return res

    def __rdivmod__(self, other):
        res = super().__rdivmod__(other)
        if self.need_grad:
            self._add_to_graph(res, 'rdivmod', other)
        return res

    def __pow__(self, other):
        res = super().__pow__(other)
        if self.need_grad:
            self._add_to_graph(res, 'pow', other)
        return res

    def __rpow__(self, other):
        res = super().__rpow__(other)
        if self.need_grad:
            self._add_to_graph(res, 'pow', other)
        return res

    @raise_err
    def __ipow__(self, other):
        return super().__ipow__(other)

    def backward(self):
        pass


if __name__ == '__main__':
    t1 = Tensor([1, 2, 3, 4])
    print(t1 + 1)
