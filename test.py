import numpy as np

import theano
import theano.tensor as tt
from theano.tests import unittest_tools as utt

from kepler_op import KeplerOp

kepler = KeplerOp()

M = tt.dmatrix()
e = tt.dmatrix()
E = kepler(M, e)

f = theano.function([M, e], E)
g = theano.function([M, e], theano.grad(tt.sum(E), [M, e]))

np.random.seed(42)
N = (10, 2)
pt = [np.random.uniform(5, 10, N), np.random.rand(*N)]

print(f(*pt))
utt.verify_grad(kepler, pt)
