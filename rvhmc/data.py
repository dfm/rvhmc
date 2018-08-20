# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["RVDataset"]

import numpy as np

import pymc3 as pm
import theano.tensor as tt

from .utils import join_names


class PolynomialTrend(object):

    def __init__(self, name, order=1, mu=None, sd=None):
        self.name = name
        self.dvdt = [
            pm.Normal(join_names(self.name, "d{0}vdt{0}".format(i+1)),
                      mu=mu, sd=sd)
            for i in range(1, self.order)
        ]

    def __call__(self, t, name=None):
        bkg = tt.zeros_like(t)
        if self.background_order > 1:
            for i in range(self.background_order):
                bkg += self.dvdt[i] * t ** (i+1)
        return pm.Deterministic(join_names(self.name, name), bkg)


class RVDataset(object):

    def __init__(self, name, t, rv, rverr, logs=None, meanrv=None, trend=None):
        self.name = name

        self.t = np.atleast_1d(t)
        self.rv = np.atleast_1d(rv)
        self.rvvar = np.atleast_1d(rverr) ** 2

        self.logs = logs
        self.trend = trend
        if meanrv is None:
            meanrv = np.mean(self.rv)
        self.meanrv = meanrv

    def observe(self, mu, name=None):
        if self.logs is None:
            sd = tt.sqrt(self.rvvar)
        else:
            sd = tt.sqrt(self.rvvar + tt.exp(2*self.logs))

        if self.meanrv is not None:
            mu = mu + self.meanrv

        if self.trend is not None:
            mu = mu + self.trend(self.t)

        return pm.Normal(join_names(self.name, name, "obs"), mu=mu, sd=sd,
                         observed=self.rv)
