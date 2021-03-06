#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["lomb_scargle_estimator", "find_peaks"]

import numpy as np
from astropy.stats import LombScargle


def lomb_scargle_estimator(x, y, yerr=None,
                           min_period=None, max_period=None,
                           filter_period=None,
                           max_peaks=2,
                           **kwargs):
    """
    Estimate period of a time series using the periodogram

    Args:
        x (ndarray[N]): The times of the observations
        y (ndarray[N]): The observations at times ``x``
        yerr (Optional[ndarray[N]]): The uncertainties on ``y``
        min_period (Optional[float]): The minimum period to consider
        max_period (Optional[float]): The maximum period to consider
        filter_period (Optional[float]): If given, use a high-pass filter to
            down-weight period longer than this
        max_peaks (Optional[int]): The maximum number of peaks to return
            (default: 2)

    Returns:
        A dictionary with the computed ``periodogram`` and the parameters for
        up to ``max_peaks`` peaks in the periodogram.

    """
    if min_period is not None:
        kwargs["maximum_frequency"] = 1.0 / min_period
    if max_period is not None:
        kwargs["minimum_frequency"] = 1.0 / max_period

    # Estimate the power spectrum
    model = LombScargle(x, y, yerr)
    freq, power = model.autopower(method="fast", normalization="psd", **kwargs)
    power /= len(x)
    power_est = np.array(power)

    # Filter long periods
    if filter_period is not None:
        freq0 = 1.0 / filter_period
        filt = 1.0 / np.sqrt(1 + (freq0 / freq) ** (2*3))
        power *= filt

    # Find and fit peaks
    peak_inds = (power[1:-1] > power[:-2]) & (power[1:-1] > power[2:])
    peak_inds = np.arange(1, len(power)-1)[peak_inds]
    peak_inds = peak_inds[np.argsort(power[peak_inds])][::-1]
    peaks = []
    for i in peak_inds[:max_peaks]:
        A = np.vander(freq[i-1:i+2], 3)
        w = np.linalg.solve(A, np.log(power[i-1:i+2]))
        sigma2 = -0.5 / w[0]
        freq0 = w[1] * sigma2
        peaks.append(dict(
            log_power=w[2] + 0.5*freq0**2 / sigma2,
            period=1.0 / freq0,
            period_uncert=np.sqrt(sigma2 / freq0**4),
        ))

    return dict(
        periodogram=(freq, power_est),
        peaks=peaks,
    )


def find_peaks(n_peaks, x, y, yerr=None, **kwargs):
    y0 = np.copy(y)

    kwargs["max_peaks"] = 1
    kwargs["samples_per_peak"] = kwargs.get("samples_per_peak", 20)

    peaks = []
    for i in range(n_peaks):
        # Find a peak using Lomb-Scargle
        m = lomb_scargle_estimator(x, y0, yerr=yerr, **kwargs)
        if not len(m):
            print("Only found {0} / {1} peaks".format(i, n_peaks))
            break
        peak = dict(m["peaks"][0])

        # Fit for the model associated with this peak
        A = np.vstack([
            np.sin(2*np.pi*x/peak["period"]),
            np.cos(2*np.pi*x/peak["period"]),
            np.ones_like(x),
        ]).T
        w = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, y0))

        # Compute the amplitude and phase
        peak["amp"] = np.sqrt(np.sum(w[:-1]**2))
        peak["phase"] = np.arctan2(w[0], w[1])

        # Subtract the best fit model
        y0 -= np.dot(A, w)

        peaks.append(peak)

    return sorted(peaks, key=lambda v: v["amp"])
