#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["RVPlanet"]

import numpy as np

import pymc3 as pm

import theano.tensor as tt

from .utils import join_names
from .transforms import UnitVector


class RVPlanet(object):

    def __init__(self, name, logP, logK, phivec=None, eccen=None,
                 omegavec=None, phi=None, omega=None):

        self.name = name

        self.logP = logP
        self.logK = logK

        if phivec is None:
            if phi is None:
                phi = 0.0
            phivec = UnitVector(join_names(self.name, "phivec"), shape=2,
                                testval=np.array([np.cos(phi), np.sin(phi)]))
        self.phivec = phivec
        self.phi = pm.Deterministic(join_names(self.name, "phi"),
                                    tt.arctan2(phivec[1], phivec[0]))

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
                if omega is None:
                    omega = 0.0
                omegavec = UnitVector(join_names(self.name, "omegavec"),
                                      shape=2,
                                      testval=np.array([np.cos(omega),
                                                        np.sin(omega)]))
            self.omegavec = omegavec
            self.omega = pm.Deterministic(join_names(self.name, "omega"),
                                          tt.arctan2(omegavec[1], omegavec[0]))

            self.vars += [self.eccen, self.omegavec]

        self.n = 2*np.pi*tt.exp(-self.logP)
        self.t0 = pm.Deterministic(join_names(self.name, "t0"),
                                   var=(self.phi + self.omega) / self.n)
