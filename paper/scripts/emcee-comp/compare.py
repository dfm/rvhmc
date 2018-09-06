#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import sys
import time
import string

import emcee
import corner
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import pymc3 as pm

import theano
import theano.tensor as tt

from rvhmc import RVDataset, PolynomialTrend, RVModel, RVPlanet


def build_model(peaks, t, y=None, yerr=None, model=None):
    model = pm.modelcontext(model)

    n_planets = len(peaks)

    if yerr is None:
        yerr = np.random.uniform(0.01, 0.1, len(t))
    if y is None:
        y = yerr*np.random.randn(len(t))

    trend = PolynomialTrend("trend", order=3)
    logs = pm.Normal("logs", mu=-5.0, sd=5.0, testval=-5.0)
    meanrv = pm.Normal("meanrv", mu=0.0, sd=10.0, testval=0.0)
    dataset = RVDataset("data", t, y, yerr, logs=logs, trend=trend,
                        meanrv=meanrv)

    logamps = pm.Uniform("logamps",
                         lower=np.log(min_amp),
                         upper=np.log(max_amp),
                         shape=n_planets,
                         testval=np.log([np.clip(peak["amp"],
                                                 min_amp+1e-2,
                                                 max_amp-1e-2)
                                         for peak in peaks]))

    planets = []
    for i, (peak, name) in enumerate(zip(peaks, string.ascii_lowercase[1:])):
        logP = pm.Uniform(name + ":logP",
                          lower=np.log(min_period),
                          upper=np.log(max_period),
                          testval=np.log(peak["period"]))
        logK = pm.Deterministic(name + ":logK", logamps[i])

        eccen = pm.Beta(name + ":eccen",
                        alpha=0.867,
                        beta=3.03,
                        testval=peak["eccen"])
        omegabase = pm.Uniform(name + ":omegabase", -2*np.pi, 2*np.pi,
                               testval=peak["omega"])
        omegavec = pm.Deterministic(name + ":omegavec",
                                    tt.stack([tt.cos(omegabase),
                                              tt.sin(omegabase)]))

        phibase = pm.Uniform(name + ":phibase", -2*np.pi, 2*np.pi,
                             testval=peak["phase"])
        phivec = pm.Deterministic(name + ":phivec",
                                  tt.stack([tt.cos(phibase), tt.sin(phibase)]))
        planets.append(
            RVPlanet(name, logP, logK, phivec=phivec, eccen=eccen,
                     omegavec=omegavec))

    rvmodel = RVModel("rv", dataset, planets)
    pm.Deterministic("logp", model.logpt)

    return rvmodel


# Simulate a random dataset
if len(sys.argv) > 1:
    n_planets = int(sys.argv[1])
else:
    n_planets = 1
dirname = "{0:02d}".format(n_planets)
if len(sys.argv) > 2:
    version = int(sys.argv[2])
    dirname = os.path.join(dirname, "{0:04d}".format(version))
else:
    version = 0
os.makedirs(dirname, exist_ok=True)

np.random.seed(42 + version)

t = np.sort(np.random.uniform(0.0, 4*365.0, 50))
yerr = np.random.uniform(0.01, 0.1, len(t))
y = yerr * np.random.randn(len(t))

min_period = 5
max_period = 100
min_amp = 0.2
max_amp = 0.8
target_n_eff = 500

peaks = []
for i in range(n_planets):
    peaks.append(dict(
        period=np.exp(np.random.uniform(np.log(min_period),
                                        np.log(max_period))),
        amp=np.exp(np.random.uniform(np.log(min_amp), np.log(max_amp))),
        phase=np.random.uniform(0, 2*np.pi),
        omega=np.random.uniform(0, 2*np.pi),
        eccen=np.random.uniform(0.01, 0.3),
    ))
peaks = sorted(peaks, key=lambda x: x["amp"])

with pm.Model() as sim_model:
    sim_rvmodel = build_model(peaks, t, y, yerr)
    f = theano.function(sim_model.vars, sim_rvmodel.get_rvmodels(t),
                        on_unused_input="ignore")
    coords = sim_model.test_point
    y += np.sum(f(*(coords[k.name] for k in sim_model.vars)), axis=1)

# Plot the data
fig = plt.figure()
plt.errorbar(t % peaks[-1]["period"], y, yerr=yerr, fmt=".k")
fig.savefig(os.path.join(dirname, "data.png"), bbox_inches="tight")
plt.close(fig)

# Work out the key variables
with pm.Model() as model:
    rvmodel = build_model(peaks, t, y, yerr)

    key_vars = [v.name for v in rvmodel.datasets[0].vars]
    key_vars += [p.name + k for p in rvmodel.planets
                 for k in (":logP", ":logK", ":phi", ":eccen", ":omega")]

# Fit using emcee
with model:
    f = theano.function(model.vars,
                        [model.logpt] + model.vars + model.deterministics)

    def log_prob_func(params):
        dct = model.bijection.rmap(params)
        args = (dct[k.name] for k in model.vars)
        results = f(*args)
        return tuple(results)

    # First we work out the shapes of all of the deterministic variables
    res = model.test_point
    vec = model.bijection.map(res)
    initial_blobs = log_prob_func(vec)[1:]
    dtype = [(var.name, float, np.shape(b)) for var, b in
             zip(model.vars + model.deterministics, initial_blobs)]

    # Then sample as usual
    coords = vec + 1e-5 * np.random.randn(3*len(vec), len(vec))
    nwalkers, ndim = coords.shape
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob_func,
                                    blobs_dtype=dtype)
    thin_by = 100
    tottime = 0
    for i in range(1000):
        strt = time.time()
        sampler.run_mcmc(coords, 50, thin_by=thin_by, progress=True)
        tottime += time.time() - strt

        samples = sampler.get_blobs()
        tau = np.array([float(emcee.autocorr.integrated_time(samples[k],
                                                             tol=0))
                        for k in key_vars])

        print(sampler.iteration * nwalkers / tau)
        converged = np.all(tau * target_n_eff / thin_by
                           < sampler.iteration * nwalkers)
        converged &= np.all(sampler.iteration > 50 * tau)
        if converged:
            break

    samples = sampler.get_blobs(discard=int(tau.max()))
    tau_emcee = np.array([float(emcee.autocorr.integrated_time(samples[k],
                                                               tol=0))
                          for k in key_vars])
    time_emcee = tottime
    time_per_emcee = time_emcee / (sampler.iteration * nwalkers)
    time_ind_emcee = time_per_emcee * tau_emcee


# Sample using pymc
with model:
    start = model.test_point

    ntune = 2000
    samples = sampler.get_chain(discard=int(tau_emcee.max()), flat=True)
    potential = pm.step_methods.hmc.quadpotential.QuadPotentialFull(
        np.cov(samples, rowvar=0))
    step = pm.NUTS(potential=potential)

#     ntune = 5000
#     _, step = pm.init_nuts(init="adapt_diag", target_accept=0.8)

    print("Running burn-in...")
    burnin = pm.sample(start=start, tune=ntune, draws=1, step=step, chains=1,
                       compute_convergence_checks=False)

    trace = None
    next_start = burnin.point(-1)
    draws = 2000
    chains = 2
    ntotal = 0
    tottime = 0
    for i in range(100):
        strt = time.time()
        trace = pm.sample(start=next_start, trace=trace, tune=0, draws=draws,
                          step=step, chains=chains,
                          compute_convergence_checks=False, cores=1)
        tottime += time.time() - strt
        ntotal += draws * chains
        next_start = [trace.point(-1, c) for c in trace.chains]

        tau = np.array([
            float(emcee.autocorr.integrated_time(np.array(
                trace.get_values(v, combine=False)).T,
                tol=0))
            for v in key_vars])
        print(tau)
        print(ntotal / tau)
        print(pm.summary(trace, varnames=key_vars).n_eff)

        if (ntotal / tau).min() > target_n_eff and ntotal > tau.max() * 50:
            break
    tau_pymc = np.copy(tau)
    time_pymc = tottime
    time_per_pymc = time_pymc / (len(trace) * chains)
    time_ind_pymc = time_per_pymc * tau_pymc

print("time per ind. sample, emcee: {0}".format(time_ind_emcee))
print("time per ind. sample, pymc: {0}".format(time_ind_pymc))
print("time per ind. sample, ratio: {0}"
      .format(time_ind_emcee / time_ind_pymc))
df = pd.DataFrame(dict(zip(key_vars, zip(time_ind_emcee, time_ind_pymc))))
df["method"] = ["emcee", "pymc"]
df.to_csv(os.path.join(dirname, "results.csv"), index=False)

tau = tau_emcee.max()
samples = sampler.get_blobs(flat=True, discard=int(2*tau), thin=int(tau))
df_emcee = pd.DataFrame.from_records(samples[key_vars])

ranges = [(np.min(df_emcee[k]), np.max(df_emcee[k])) for k in df_emcee.columns]

df_pymc = pm.trace_to_dataframe(trace, varnames=key_vars)

w_pymc = len(df_emcee) / len(df_pymc) + np.zeros(len(df_pymc))

v = key_vars[:15]
fig = corner.corner(df_emcee[v], color="C0",
                    range=ranges[:len(v)])
corner.corner(df_pymc[v], weights=w_pymc, color="C1", fig=fig,
              range=ranges[:len(v)])
fig.savefig(os.path.join(dirname, "corner.png"), bbox_inches="tight")
plt.close(fig)
