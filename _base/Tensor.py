#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import numpy.lib.user_array as UsrArry
# def get_cpt_func(name, grad=False):
#     if not grad:
#         funcs = {
#             'add': (lambda x, y: x+y),
#             'multi':  (lambda x, y: x*y),
#             'divide': (lambda x, y: x/y),
#             'dot': (lambda x, y: x.dot(y)),
#             'sigmoid': (lambda x: 1 / (1+(-x).exp()))
#         }
#     else:
#         funcs = {
#             'add': (lambda x, y: x+y),
#             'multi':  (lambda x, y: x*y),   
#             'divide': (lambda x, y: x/y),
#             'dot': (lambda x, y: x.dot(y)),
#             'sigmoid': (lambda x: 1 / (1+(-x).exp())),
#         }
#     return funcs[name]

# def type_limited(func):
#     '''
#     limit the inputs's type of the method of a `Tensor`.
#     '''

#     def wrapper(self, inputs):
#         if not isinstance(inputs, Tensor):
#             raise TypeError(
#                 'Input data should be `Tensor`, not {}'.format(type(inputs)))
#         return func(self, inputs)
#     return wrapper

# class CptNode(object):
#     def __init__(self, inputs, fwd_func, bak_func, upper_grad, name):
#         '''
#         The node of a computational graph, representing a kind of computation.
#         paramters:

#         - inputs: Tensor(s) fed to the current node
#         - fwd_fun: the forward computation of the current node
#         - bak_func: derivative of the current node, used to backproping.
#         - upper_grad: the grad from the upper nodes.
#         - name: the name of the current node, choosed from the dict `name2func`.
#         '''

#         self.inputs = inputs
#         self.fwd_func = fwd_func
#         self.bak_func = bak_func
#         self.upper_grad = upper_grad
#         self.name = name

#     def forward(self):
#         self.output = self.fwd_func(self.inputs)
#         return self.output

#     def cal_grad(self, upper_grad):
#         return self.bak_func(self.grad) * upper_grad

#     def __str__(self):
#         return 'ComputationNode: {}, input:{}, output:{}'.format(self.name, self.inputs, self.output)


class Tensor(UsrArry.container):
    def __init__(self, data, dtype=None, copy=True, need_grad=False, upper_nodes=None, bak_nodes=None, grad=None, name=None):
        '''
        paramsters:

        - data: must be a list or other iterable object.
        - upper_node: the upper computational node of this Tensor.
        - back_node: the computational node in the downstream.
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

        self.need_grad = need_grad
        # self.cpt_func = get_cpt_func(name)
        # self.grad_func = get_cpt_func(name, grad=True)
        self.grad = np.ones(self.data.shape) if self.need_grad else None
        self.upper_nodes = upper_nodes
        self.bak_nodes = bak_nodes




    def backward(self):
        reader = self.bak_nodes
        while 1: 
            for x in reader.bak_nodes:
                
                reader = reader.bak_nodes
        
    


if __name__ == '__main__':
    t1 = Tensor([1, 2, 3, 4])
    t2 = Tensor([2, 3, 4, 5])
    print(t1+t2)
