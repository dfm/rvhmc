# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["RVDataset"]

import numpy as np

import pymc3 as pm
import theano.tensor as tt

from .utils import join_names


class RVDataset(object):

    def __init__(self, t, rv, rverr,
                 logs=0.0, logs_range=None,
                 background_order=1, meanrv=None, priorsd=None, name=None):
        self.name = name

        self.t = np.atleast_1d(t)
        self.rv = np.atleast_1d(rv)
        self.rvvar = np.atleast_1d(rverr) ** 2

        if logs_range is None:
            self.logs = pm.Flat(join_names(self.name, "logs"), testval=logs)
        else:
            self.logs = pm.Uniform(join_names(self.name, "logs"),
                                   lower=logs_range[0],
                                   upper=logs_range[1],
                                   testval=logs)

        self.background_order = background_order

        # Background model
        if meanrv is None:
            meanrv = np.mean(self.rv)
        if priorsd is None:
            priorsd = np.max(self.rv) - np.min(self.rv)
        self.meanrv = pm.Normal(join_names(self.name, "meanrv"),
                                mu=meanrv, sd=priorsd,
                                testval=meanrv)
        if self.background_order > 1:
            self.dvdt = [
                pm.Normal(join_names(self.name, "d{0}vdt{0}".format(i+1)),
                          mu=0.0, sd=priorsd, testval=0.0)
                for i in range(self.background_order)
            ]

    def get_background(self, t, name=None):
        bkg = self.meanrv + tt.zeros_like(t)
        if self.background_order > 1:
            for i in range(self.background_order):
                bkg += self.dvdt[i] * t ** (i+1)
        return pm.Deterministic(join_names(self.name, name, "bkg"), bkg)

    def observe(self, mu, name=None):
        sd = tt.sqrt(self.rvvar + tt.exp(2*self.logs))
        return pm.Normal(join_names(self.name, name, "obs"),
                         mu=mu, sd=sd, observed=self.rv)
