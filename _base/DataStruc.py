#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np

class Tensor(object):
    def __init__(self, mat):
        r'''
        - mat: a list or other iterable object.
        '''
        self.data = np.array(mat)
        

    def __str__(self):
        return self.data.__str__(), 'type={}, {}'.format(self.data.dtype, self.data.shape)

    