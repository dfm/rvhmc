# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["setup_default_model"]

import string
import collections

import numpy as np

import pymc3 as pm
from pymc3.model import modelcontext

import theano.tensor as tt

from .rvmodel import RVModel
from .rvplanet import RVPlanet

from .estimator import find_peaks


def setup_default_model(n_planets, datasets,
                        min_period=None, max_period=None,
                        min_amp=None, max_amp=None,
                        circular=True,
                        trend_order=0,
                        model=None):
    model = modelcontext(model)

    if isinstance(datasets, collections.Iterable):
        datasets = datasets
    else:
        datasets = [datasets]

    x, y, yerr = [], [], []
    for data in datasets:
        x.append(data.t)
        y.append(data.rv)
        if data.rverr is not None:
            yerr.append(data.rverr)
    x = np.concatenate(x)
    y = np.concatenate(y)
    if len(yerr):
        yerr = np.concatenate(yerr)
        if len(yerr) != len(x):
            yerr = None
    else:
        yerr = None

    if min_period is None:
        min_period = np.mean(np.diff(np.sort(x)))
    if max_period is None:
        max_period = 0.5*(x.max() - x.min())

    if min_amp is None:
        if yerr is None:
            min_amp = 0.001 * np.std(y)
        else:
            min_amp = 0.01 * np.min(yerr)
    if max_amp is None:
        max_amp = 1.5 * (y.max() - y.min())

    peaks = find_peaks(n_planets, x, y, yerr,
                       min_period=min_period, max_period=max_period)

    with model:
        planets = []
        for peak, name in zip(peaks, string.ascii_lowercase[1:]):
            logP = pm.Uniform(name + ":logP",
                              lower=np.log(min_period),
                              upper=np.log(max_period),
                              testval=np.log(peak["period"]))
            logK = pm.Uniform(name + ":logK",
                              lower=np.log(min_amp),
                              upper=np.log(max_amp),
                              testval=np.log(np.clip(peak["amp"],
                                                     min_amp+1e-2,
                                                     max_amp-1e-2)))

            eccen = None
            if not circular:
                eccen = pm.Beta(name + ":eccen",
                                alpha=0.867,
                                beta=3.03,
                                testval=0.001)

            planets.append(
                RVPlanet(name, logP, logK, phi=peak["phase"], eccen=eccen))

            if len(planets) > 1:
                pm.Potential("order:{0}".format(name),
                             tt.switch((planets[-2].logK < planets[-1].logK),
                                       0.0, -np.inf))

        rvmodel = RVModel("rv", datasets, planets)
        pm.Deterministic("logp", model.logpt)

        return rvmodel
