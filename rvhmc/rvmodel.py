# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["RVModel"]

import collections

import pymc3 as pm
from pymc3.model import modelcontext

import theano.tensor as tt

from .utils import join_names
from .kepler_op import KeplerOp


class RVModel(object):

    def __init__(self, name, datasets, planets, tol=1e-8, maxiter=2000):
        self.name = name

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
        return pm.Deterministic(join_names(self.name, name, "rvmodels"),
                                K * (cosw*(tt.cos(f)+e) - sinw*tt.sin(f)))

    def observe(self, name=None):
        for data in self.datasets:
            model = pm.Deterministic(
                join_names(self.name, data.name, name, "model"),
                tt.sum(self.get_rvmodels(
                    data.t, join_names(data.name, name)),
                       axis=1))
            data.observe(model, name)

    def optimize(self, start=None, vars=None, model=None, **kwargs):
        model = modelcontext(model)

        if start is None:
            start = model.test_point

        soln = pm.find_MAP(start, vars=vars, **kwargs)

        old_logp = model.logp(start)
        new_logp = model.logp(soln)

        if new_logp > old_logp:
            return soln

        return start

    def initialize(self, start=None, model=None, **kwargs):
        model = modelcontext(model)
        if start is None:
            start = model.test_point

        print("Initial logp: {0}".format(model.logp(start)))

        print("Optimizing phases...")
        soln = self.optimize(vars=[p.phivec for p in self.planets], **kwargs)

        print("Optimizing planet parameters...")
        for planet in self.planets:
            if not len(planet.vars):
                continue
            soln = self.optimize(start=soln, vars=planet.vars, **kwargs)

        print("Optimizing dataset parameters...")
        for data in self.datasets:
            if not len(data.vars):
                continue
            soln = self.optimize(start=soln, vars=data.vars, **kwargs)

        print("Optimizing all planet parameters...")
        soln = self.optimize(start=soln,
                             vars=[v for p in self.planets for v in p.vars],
                             **kwargs)

        print("Optimizing all parameters...")
        soln = self.optimize(start=soln, **kwargs)

        print("Final logp: {0}".format(model.logp(soln)))

        return soln
