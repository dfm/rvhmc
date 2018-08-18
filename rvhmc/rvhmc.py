# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["RVModel", "RVPlanet"]

import collections

import numpy as np

import pymc3 as pm
import pymc3.distributions.transforms as tr

import theano.tensor as tt

from .utils import join_names
from .kepler_op import KeplerOp


class UnitVector(tr.Transform):
    name = "unit_vector"

    def backward(self, y):
        norm = tt.sqrt(tt.sum(tt.square(y), axis=-1, keepdims=True))
        return y / norm

    def forward(self, x):
        return tt.as_tensor_variable(x)

    def forward_val(self, x, point=None):
        return np.copy(x)

    def jacobian_det(self, y):
        return -0.5*tt.sum(tt.square(y), axis=-1)


unit_vector = UnitVector()


# def unit_vector(name, shape=None, testval=None):
#     vec = pm.Normal(name + "_latent__", shape=shape, testval=testval)
#     norm = tt.sqrt(tt.sum(tt.square(vec), axis=-1, keepdims=True))
#     return vec/norm


class RVModel(object):

    def __init__(self, datasets, planets,
                 tol=1e-8, maxiter=2000):

        if isinstance(datasets, collections.Iterable):
            self.datasets = datasets
        else:
            self.datasets = [datasets]

        if isinstance(planets, collections.Iterable):
            self.planets = planets
        else:
            self.planets = [planets]

        self.kepler = KeplerOp(tol=tol, maxiter=maxiter)

        self.observe()

    def get_rvmodels(self, t, name=None):
        K = tt.exp(tt.stack([p.logK for p in self.planets]))
        n = tt.stack([p.n for p in self.planets])
        t0 = tt.stack([p.t0 for p in self.planets])
        e = tt.stack([p.eccen for p in self.planets])
        cosw = tt.stack([p.omegavec[0] for p in self.planets])
        sinw = tt.stack([p.omegavec[1] for p in self.planets])

        mean_anom = n * (t[:, None] - t0)
        eccen_arg = e + tt.zeros_like(mean_anom)
        eccen_anom = self.kepler(mean_anom, eccen_arg)
        f = 2*tt.arctan2(tt.sqrt(1+e)*tt.tan(0.5*eccen_anom),
                         tt.sqrt(1-e)+tt.zeros_like(eccen_anom))
        return pm.Deterministic(join_names(name, "rvmodels"),
                                K * (cosw*(tt.cos(f)+e) - sinw*tt.sin(f)))

    def observe(self, name=None):
        for data in self.datasets:
            model = pm.Deterministic(
                join_names(data.name, name, "model"),
                tt.sum(self.get_rvmodels(data.t, join_names(data.name, name)),
                       axis=1))
            data.observe(model, name)


class RVPlanet(object):

    def __init__(self,
                 name,
                 logP,
                 logK,
                 phivec=None,
                 eccen=None,
                 omegavec=None,
                 logP_range=None,
                 logK_range=None):

        self.name = name

        self.logP = logP
        self.logK = logK

        self._angle("omega", phivec)

        self.vars = [self.logP, self.logK, self.phivec]

        if eccen is None:
            self.circular = True
            self.eccen = 0.0
            self.omegavec = np.array([1.0, 0.0])
            self.omega = 0.0
        else:
            self.circular = False

            self.eccen = eccen

            if omegavec is None:
                omegavec = self.get_omegavec(np.array([1.0, 0.0]))
            self.omegavec = omegavec
            self.omega = pm.Deterministic(
                join_names(self.name, "omega"),
                tt.arctan2(self.omegavec[1], self.omegavec[0]))

            self.vars += [self.eccen, self.omegavec]

        self.n = 2*np.pi*tt.exp(-self.logP)
        self.t0 = pm.Deterministic(join_names(self.name, "t0"),
                                   var=(self.phi + self.omega) / self.n)

    def _angle(self, name, testval=np.array([1.0, 0.0])):
        vec = pm.Flat(join_names(self.name, name + "vec"), shape=2,
                      transform=unit_vector, testval=testval)
        setattr(self, name + "vec", vec)
        setattr(self, name, pm.Deterministic(
            join_names(self.name, name), tt.arctan2(vec[1], vec[0])))

    def get_phivec(self, testval=None):
        return pm.Flat(join_names(self.name, "phivec"), shape=2,
                       transform=unit_vector, testval=testval)

    def get_omegavec(self, testval=None):
        return pm.Flat(join_names(self.name, "omegavec"), shape=2,
                       transform=unit_vector, testval=testval)

    def get_eccen(self, testval=None):
        return pm.Uniform(join_names(self.name, "eccen"), lower=0.0, upper=1.0)
