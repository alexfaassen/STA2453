import numpy as np

def rpareto(n, xm=1.0, alpha=1.0, offset=0.0, upper=np.inf):
    """
    Draw 'n' samples from a Pareto distribution with scale xm and shape alpha,
    shift them by 'offset', and reject any above 'upper'.

    Equivalent to the R version:
       x_temp <- xm / (U^(1/alpha)) + offset
       keep only x_temp <= upper
    """
    samples = []
    while len(samples) < n:
        needed = n - len(samples)
        # uniform(0, 1)
        u = np.random.random(needed)
        x_temp = xm / (u ** (1 / alpha)) + offset
        # filter by 'upper'
        x_temp = x_temp[x_temp <= upper]
        # append
        samples.append(x_temp)
    return np.concatenate(samples)[:n]


def rmovexp(n, rate=1.0, offset=0.0):
    """
    Return 'n' samples from an exponential with 'rate' (1/scale),
    shifted by 'offset'.
    """
    scale = 1.0 / rate
    return offset + np.random.exponential(scale, size=n)


def kepler_raising(x):
    """
    Kepler flare RISE phase polynomial from Davenport et al. (2014).
    Expects x in [ -1, 0 ] when scaled by t_half.
    """
    return (1
            + 1.941 * x
            - 0.175 * x**2
            - 2.246 * x**3
            - 1.125 * x**4)


def kepler_decay(x):
    """
    Kepler flare DECAY phase from Davenport et al. (2014).
    Modeled as the sum of two exponentials.
    """
    return (0.6890 * np.exp(-1.6 * x)
            + 0.3030 * np.exp(-0.2783 * x))


def kepler_flare(tt, t_half, n,
                 flux_dist=rpareto,
                 **dist_kwargs):
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
    states : ndarray
        An integer array indicating the “state” at each time:
          1 => baseline,
          2 => within a flare’s rise,
          3 => within a flare’s decay.
    """
    tt = np.asarray(tt)
    flare = np.zeros_like(tt, dtype=float)
    states = np.ones_like(tt, dtype=int)  # baseline=1

    # Draw random flare amplitudes
    flux_all = flux_dist(n=n, **dist_kwargs)

    # Choose random times for the flare peaks
    # R's sample(tt, n) is sampling w/o replacement by default.
    # For large n < len(tt), that may matter, but we can do the same:
    peak_time_all = np.random.choice(tt, size=n, replace=False)
    # Sort them if you want them in ascending order:
    peak_time_all.sort()

    # Loop through each flare
    for i in range(n):
        flux_loc = flux_all[i]
        # scale t_half by flux to produce a proportionally wider flare
        t_half_loc = t_half * flux_loc
        peak_time_loc = peak_time_all[i]

        # Indices for the rise phase
        #  rise:   peak_time - t in [0, t_half_loc]
        #  => 0 <= peak_time_loc - t <= t_half_loc
        #  => peak_time_loc - t_half_loc <= t <= peak_time_loc
        rising_phase = np.where(
            (tt <= peak_time_loc) &
            (tt >= peak_time_loc - t_half_loc)
        )[0]

        # Indices for the decay phase
        #  decay:  t - peak_time in [0, 10 * t_half_loc]
        #  => 0 <= t - peak_time_loc <= 10 * t_half_loc
        #  => peak_time_loc <= t <= peak_time_loc + 10*t_half_loc
        decaying_phase = np.where(
            (tt >= peak_time_loc) &
            (tt <= peak_time_loc + 10.0 * t_half_loc)
        )[0]

        # Tag the states
        states[rising_phase] = 2
        states[decaying_phase] = 3

        # Compute scaled times within the rise/decay windows
        raise_time_loc = (tt[rising_phase] - peak_time_loc) / t_half_loc
        decay_time_loc = (tt[decaying_phase] - peak_time_loc) / t_half_loc

        # Evaluate the flare shape, multiplied by flux_loc
        # (Note that for the rise phase we expect negative scaled times.)
        raise_flux = flux_loc * kepler_raising(raise_time_loc)
        decay_flux = flux_loc * kepler_decay(decay_time_loc)

        # Add them into the 'flare' array
        flare[rising_phase] += raise_flux
        flare[decaying_phase] += decay_flux

    return flare, states