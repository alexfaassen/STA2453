"""simulate_flare.py

Generates synthetic stellar flares based on the Kepler flare model
(Davenport et al., 2014). Includes functions for custom random sampling
and flare injection using rise- and decay- phase polynomials.
"""

import numpy as np
from typing import Callable, Tuple, List

def rpareto(
    n: int,
    xm: float = 1.0,
    alpha: float = 1.0,
    offset: float = 0.0,
    upper: float = np.inf
) -> np.ndarray:
    """
    Draw 'n' samples from a Pareto distribution with scale xm and shape alpha,
    shift them by 'offset', and reject any above 'upper'.
    """
    samples = []
    while len(samples) < n:
        needed = n - len(samples)
        # Uniform(0, 1)
        u = np.random.random(needed)
        x_temp = xm / (u ** (1 / alpha)) + offset
        # Upper filter
        x_temp = x_temp[x_temp <= upper]

        samples.append(x_temp)
    return np.concatenate(samples)[:n]


def rmovexp(n: int, rate: float = 1.0, offset: float = 0.0) -> np.ndarray:
    """
    Return 'n' samples from an exponential with 'rate' (1/scale),
    shifted by 'offset'.
    """
    scale = 1.0 / rate
    return offset + np.random.exponential(scale, size=n)


def kepler_raising(x: np.ndarray) -> np.ndarray:
    """
    Kepler flare RISE phase polynomial from Davenport et al. (2014).
    Expects x in [ -1, 0 ] when scaled by t_half.
    """
    return (1
            + 1.941 * x
            - 0.175 * x**2
            - 2.246 * x**3
            - 1.125 * x**4)


def kepler_decay(x: np.ndarray) -> np.ndarray:
    """
    Kepler flare DECAY phase from Davenport et al. (2014).
    Modeled as the sum of two exponentials.
    """
    return (0.6890 * np.exp(-1.6 * x)
            + 0.3030 * np.exp(-0.2783 * x))


def kepler_flare(
    tt: np.ndarray,
    t_half: float,
    n: int,
    flux_dist: Callable[..., np.ndarray] = rpareto,
    **dist_kwargs
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Generate 'n' flares sampled in time over the array 'tt'.
    Each flare uses the classical Kepler-rise and Kepler-decay shape,
    scaled by a random flux amplitude drawn from 'flux_dist'.

    Parameters
    ----------
    tt : array-like
        Array of time values (e.g. np.linspace) over which to simulate.
    t_half : float
        Base timescale for the half-peak width. Actual half-peak for each flare
        scales with the individual flare flux amplitude.
    n : int
        Number of flares to simulate.
    flux_dist : callable, optional
        Distribution function returning random amplitudes for each flare.
        Defaults to Pareto as in the example R code.
    dist_kwargs : dict
        Extra keyword arguments passed to the 'flux_dist' sampler
        (e.g. alpha, xm, offset).

    Returns
    -------
    flare : ndarray
        The synthetic flare light curve (same length as 'tt').
    flare_times: List[Tuple[int, int]]
        List of (start_idx, end_idx) for each flare.
    """

    ## Initialize
    tt = np.asarray(tt)
    flare = np.zeros_like(tt, dtype=float)
    flare_times = []

    ## Flare amplitudes
    flare_amps = flux_dist(n=n, **dist_kwargs) # Pareto distribution

    ## Flare peak times (SRSWOR)
    peak_times = np.random.choice(tt, size=n, replace=False)
    peak_times.sort()

    ## Iterate: Create flares
    for i in range(n):
        flare_amp = flare_amps[i]
        # scale t_half by flux to produce a proportionally wider flare
        t_half_loc = t_half * flare_amp # Flare rise interval length
        peak_time_loc = peak_times[i] # Peak time

        # Rising phase: peak_time - t_half_i to peak_time
        rising_phase = np.where(
            (tt <= peak_time_loc) &
            (tt >= peak_time_loc - t_half_loc)
        )[0]

        # Decay phase: peak_time to peak_time + 10 * t_half_i
        decaying_phase = np.where(
            (tt >= peak_time_loc) &
            (tt <= peak_time_loc + 10.0 * t_half_loc)
        )[0]

        # Scaled times
        raise_time_loc = (tt[rising_phase] - peak_time_loc) / t_half_loc
        decay_time_loc = (tt[decaying_phase] - peak_time_loc) / t_half_loc

        # Flare shape
        raise_flux = flare_amp * kepler_raising(raise_time_loc)
        decay_flux = flare_amp * kepler_decay(decay_time_loc)

        flare[rising_phase] += raise_flux
        flare[decaying_phase] += decay_flux

        # Start/end times
        if len(rising_phase) > 0 and len(decaying_phase) > 0:
            start_i = rising_phase[0]
            end_i = decaying_phase[-1]
            flare_times.append((start_i, end_i))

    return flare, flare_times