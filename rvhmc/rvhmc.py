# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["KeplerRVModel"]

import numpy as np

import pymc3 as pm

import theano
import theano.tensor as tt


class KeplerRVModel(object):

    def __init__(self, data, planets, bkg_order=3, log_s2=None):
        self.data = data
        self.bkg_order = bkg_order
        self.log_s2 = log_s2


class KeplerRVPlanet(object):
    pass
