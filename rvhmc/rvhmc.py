# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["kepler"]

import os
import sysconfig
import tensorflow as tf


# Load the ops library
suffix = sysconfig.get_config_var("EXT_SUFFIX")
dirname = os.path.dirname(os.path.abspath(__file__))
libfile = os.path.join(dirname, "kepler_op")
if suffix is not None:
    libfile += suffix
else:
    libfile += ".so"
custom_ops = tf.load_op_library(libfile)


# ------
# Kepler
# ------

def kepler(M, e, **kwargs):
    """Solve Kepler's equation

    Args:
        M: A Tensor of mean anomaly values.
        e: The eccentricity (this must be a scalar).
        maxiter (Optional): The maximum number of iterations to run.
        tol (Optional): The convergence tolerance.

    Returns:
        A Tensor with the eccentric anomaly evaluated for each entry in ``M``.

    """
    return custom_ops.kepler(M, e, **kwargs)


@tf.RegisterGradient("Kepler")
def _kepler_grad(op, *grads):
    M, e = op.inputs
    E = op.outputs[0]
    bE = grads[0]
    bM = bE / (1.0 - e * tf.cos(E))
    be = tf.sin(E) * bM
    # be = tf.reduce_sum(tf.sin(E) * bM)
    return [bM, be]
