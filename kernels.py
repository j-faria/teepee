# -*- coding: utf-8 -*-
import numpy as np

from sys import float_info
eps = float_info.epsilon


class Kernel(object):
    is_teepee_kernel = True

    def __init__(self, *pars):
        self.pars = np.array(pars)

    def __call__(self, x1, x2=None):
        raise NotImplementedError, 'instances should implement __call__() !'

    def value(self, x1, x2=None):
        return self.__call__(x1, x2)

    def __add__(self, b):
        if not hasattr(b, "is_teepee_kernel"):
            b = ConstantKernel(float(b))
        return Sum(b, self)

    def __radd__(self, b):
        return self.__add__(b)

    def __mul__(self, b):
        if not hasattr(b, "is_teepee_kernel"):
            b = ConstantKernel(float(b))
        return Product(b, self)

    def __rmul__(self, b):
        return self.__mul__(b)

    def __repr__(self):
        """ Text representation of each Kernel instance """
        return "{0}({1})".format(self.__class__.__name__,
                                 ", ".join(map(str, self.pars)))


class _operator(Kernel):
    is_teepee_kernel = False
    def __init__(self, k1, k2):
        self.k1 = k1
        self.k2 = k2

    @property
    def pars(self):
        return np.append(self.k1.pars, self.k2.pars)


class Sum(_operator):
    """ Sum of kernels """
    def __repr__(self):
        return "{0} + {1}".format(self.k1, self.k2)

    def __call__(self, r):
        return self.k1(r) + self.k2(r)


class Product(_operator):
    """ Multplication of kernels """
    def __repr__(self):
        return "{0} * {1}".format(self.k1, self.k2)
        
    def __call__(self, r):
        return self.k1(r) * self.k2(r)





class ConstantKernel(Kernel):
    """ 
    This kernel returns a constant value over 
    the whole covariance matrix.
    """
    def __init__(self, value):
        super(ConstantKernel, self).__init__(value)
    
    def __call__(self, x1, x2=None):
        if x2 is None:
            x2 = x1
        return self.pars[0] * np.ones((x1.shape[0], x2.shape[0]))


class WhiteKernel(Kernel):
    def __init__(self, value):
        super(WhiteKernel, self).__init__(value)

    def __call__(self, x1, x2=None):
        if x2 is None:
            return self.pars[0] * np.identity(x1.size)
        else:
            # 10*eps is necessary because of floating-point reasons
            # which means this implementation is not the best
            # also it is not incredibly fast for big arrays
            return self.pars[0] * \
                   (np.abs(np.subtract.outer(x1,x2).squeeze()) <= 10*eps).astype(np.float)


class ExpSquaredKernel(Kernel):
    def __init__(self, length_scale):
        super(ExpSquaredKernel, self).__init__(length_scale)
    
    def __call__(self, x1, x2=None):
        if x2 is None:
            x2 = x1

        # x1[:, None] - x2[None, :] seems slightly faster but
        # np.subtract.outer might be a bit more readable...
        return np.exp(-0.5*self.pars[0]*np.subtract.outer(x1, x2)**2)

