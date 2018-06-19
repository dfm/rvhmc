#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import time
import numpy as np
import tensorflow as tf

import tfmodeling as tfm

import rvhmc

session = tf.InteractiveSession()

np.random.seed(1234)

N_pl = 2

T = tf.float64

# Data
t = np.sort(np.random.uniform(0, 365.0, 150))
yerr = np.random.uniform(1, 5, len(t))
y = yerr * np.random.randn(len(t))
t_tensor = tf.placeholder(T, name="t")
feed_dict = {t_tensor: t}

# Parameters
lk = np.log(np.random.uniform(5, 15, N_pl))
lk[0] = -4
log_K = tfm.Parameter(lk, bounds=(-10, 10), name="log_K", dtype=T)
log_P = tfm.Parameter(np.random.uniform(np.log(5), np.log(50), N_pl),
                      bounds=(np.log(5), np.log(100)), name="log_P", dtype=T)
omega_vec = tfm.UnitVector(np.random.randn(N_pl, 2), name="omega_vec", dtype=T)
e = tfm.Parameter(np.random.uniform(0, 0.1, N_pl), bounds=(0, 1), name="e",
                  dtype=T)
phi_vec = tfm.UnitVector(np.random.randn(N_pl, 2), name="phi_vec", dtype=T)
rv0 = tfm.Parameter(0.0, bounds=(-500, 500), name="rv0", dtype=T)
log_jitter = tfm.Parameter(-10, bounds=(-15, 0.0), name="log_jitter", dtype=T)

# Parameter transformations
K = tf.exp(log_K.value)
P = tf.exp(log_P.value)
sin_omega = omega_vec.value[:, 0]
cos_omega = omega_vec.value[:, 1]
omega = tf.atan2(sin_omega, cos_omega)
phi = tf.atan2(phi_vec.value[:, 0], phi_vec.value[:, 1])
jitter2 = tf.exp(2*log_jitter.value)

# The RV model
n = 2.0 * np.pi / P
t0 = (phi + omega) / n
M = n * t_tensor[:, None] - (phi + omega)
E = rvhmc.kepler(M, e.value + tf.zeros_like(M))
f = 2*tf.atan2(tf.sqrt(1+e.value)*tf.tan(0.5*E),
               tf.sqrt(1-e.value)+tf.zeros_like(E))
rv_models = K * (cos_omega*(tf.cos(f)+e.value) - sin_omega*tf.sin(f))

# Sum the contributions from each planet
rv = rv0.value + tf.reduce_sum(rv_models, axis=1)

session.run(tf.global_variables_initializer())

# Simulate the data
y += rv.eval(feed_dict=feed_dict)

# Compute the likelihood
yerr2 = yerr**2
log_like = -0.5 * tf.reduce_sum(
    tf.square(y - rv) / (yerr2 + jitter2) + tf.log(yerr2 + jitter2)
)

params = [log_K, log_P, omega_vec, e, phi_vec, rv0, log_jitter]
model = tfm.Model(log_like, params, feed_dict={t_tensor: t})
true_params = model.current_vector()

vec = model.current_vector()
model.value(vec)

K = 500
strt = time.time()
for k in range(K):
    model.value(vec)
print((time.time() - strt) / K)

vec = model.current_vector()
model.gradient(vec)

K = 500
strt = time.time()
for k in range(K):
    model.gradient(vec)
print((time.time() - strt) / K)
