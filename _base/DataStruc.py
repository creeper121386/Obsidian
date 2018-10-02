#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


def type_limit(func):
    '''
    to limit the inputs's type of Tensor's method.
    '''
    def wrapper(self, inputs):
        if not isinstance(inputs, Tensor):
            raise TypeError(
                'Input data should be `Tensor`, not {}'.format(type(inputs)))
        return func(self, inputs)
    return wrapper


class Tensor(object):
    def __init__(self, data, in_graph=False, need_grad=False, upper_node=None, back_node=None, grad=None):
        
        '''
        paramsters:

        - data: must be a list or other iterable object.
        - in_graph: if this Tensor is in a computational graph.
        - upper_node: the upper computational node of this Tensor.
        - back_node: the computational node in the downstream.
        - grad: the grad of this This node.
        - need_grad: need grad or not.
        '''

        self.data = np.array(data)
        self.is_node = False
        self.in_graph = in_graph

        if self.in_graph:
            if self.need_grad:
                self.grad = np.zeros(self.data.shape)
            self.upper_node = upper_node
            self.back_node = back_node

    def __str__(self):
        return self.data.__str__() + ', type={}, size={}\n'.format(self.data.dtype, self.data.shape)
        

    def __getitem__(self, ix):
        return self.data[ix]

    @type_limit
    def dot(self, t):
        return self.data.dot(t.data)


class CptNode(object):
    def __init__(self, inputs, fwd_func, bak_func, upper_grad, name):
        '''
        The node of a computational graph, representing a kind of computation.
        paramters:

        - inputs: Tensor(s) fed to the current node
        - fwd_fun: the forward computation of the current node
        - bak_func: derivative of the current node, used to backproping.
        - upper_grad: the grad from the upper nodes.
        - name: the name of the current node, choosed from the dict `name2func`.
        '''

        self.inputs = inputs
        self.fwd_func = fwd_func
        self.bak_func = bak_func
        self.upper_grad = upper_grad
        self.name = name

    def forward(self):
        self.output = self.fwd_func(self.inputs)
        return self.output

    def cal_grad(self, upper_grad):
        return self.bak_func(self.grad) * upper_grad

    def __str__(self):
        return 'ComputationNode: {}, input:{}, output:{}'.format(self.name, self.inputs, self.output)


class Graph(object):
    def __init__(self, Graph_dict):
        self.struc = Graph_dict


if __name__ == '__main__':
    t1 = Tensor([1, 2, 3, 4])
    t1.upper_node = Tensor([5, 6, 7, 8])
    t2 = t1.upper_node
    t1.upper_node = 0

    print('t2=', t2, '\nt1.upper=', t1.upper_node)
