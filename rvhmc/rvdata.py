# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["PolynomialTrend", "RVDataset", "FixedGPRVDataset"]

import numpy as np
from scipy.linalg import cho_solve

import pymc3 as pm
import theano.tensor as tt

from .utils import join_names


class PolynomialTrend(object):

    def __init__(self, name, order=1, mu=0.0, sd=1.0):
        self.name = name
        self.order = order
        self.vars = [
            pm.Normal(join_names(self.name, "d{0}vdt{0}".format(i)),
                      mu=mu, sd=sd)
            for i in range(1, self.order)
        ]

    def __call__(self, t, name=None):
        bkg = tt.zeros(len(t))
        if self.order > 1:
            for i in range(self.order-1):
                bkg += self.vars[i] * t ** (i+1)
        return pm.Deterministic(join_names(self.name, name), bkg)


class RVDataset(object):

    def __init__(self, name, t, rv, rverr=None, logs=None, meanrv=None,
                 trend=None):
        self.name = name
        self.vars = []

        self.t = np.atleast_1d(t)
        self.rv = np.atleast_1d(rv)
        self.rverr = rverr
        if rverr is not None:
            self.rvvar = np.atleast_1d(rverr) ** 2

        self.logs = logs
        if self.logs is not None:
            self.vars.append(self.logs)

        self.trend = trend
        if self.trend is not None:
            self.vars += self.trend.vars

        if meanrv is None:
            meanrv = np.mean(self.rv)
        else:
            self.vars.append(meanrv)
        self.meanrv = meanrv

    def observe(self, mu, name=None):
        if self.logs is None:
            sd = tt.sqrt(self.rvvar)
        elif self.rverr is None:
            sd = tt.exp(self.logs)
        else:
            sd = tt.sqrt(self.rvvar + tt.exp(2*self.logs))

        if self.meanrv is not None:
            mu = mu + self.meanrv

        if self.trend is not None:
            mu = mu + self.trend(self.t)

        return pm.Normal(join_names(self.name, name, "obs"), mu=mu, sd=sd,
                         observed=self.rv)


class FixedGPRVDataset(RVDataset):

    def __init__(self, name, t, rv, rverr, K, meanrv=None, trend=None):
        self.cholesky_L = np.linalg.cholesky(K + np.diag(rverr**2))
        self.L_inv_K_T = cho_solve((self.cholesky_L, True), K).T
        super(FixedGPRVDataset, self).__init__(name, t, rv, rverr,
                                               meanrv=meanrv,
                                               trend=trend)

    def observe(self, mu, name=None):
        if self.meanrv is not None:
            mu = mu + self.meanrv

        if self.trend is not None:
            mu = mu + self.trend(self.t)

        resid = self.rv - mu
        alpha = tt.slinalg.solve_lower_triangular(self.cholesky_L, resid)

        pm.Deterministic(join_names(self.name, name, "gp"),
                         tt.dot(self.L_inv_K_T, alpha))

        return pm.Potential(join_names(self.name, name, "obs"),
                            -0.5*tt.sum(alpha**2))

        # return pm.MvNormal(join_names(self.name, name, "obs"),
        #                    mu=mu, chol=self.cholesky_L,
        #                    observed=self.rv)
