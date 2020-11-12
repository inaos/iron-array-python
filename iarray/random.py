###########################################################################################
# Copyright INAOS GmbH, Thalwil, 2018.
# Copyright Francesc Alted, 2018.
#
# All rights reserved.
#
# This software is the confidential and proprietary information of INAOS GmbH
# and Francesc Alted ("Confidential Information"). You shall not disclose such Confidential
# Information and shall use it only in accordance with the terms of the license agreement.
###########################################################################################

import iarray as ia
from iarray import iarray_ext as ext


def random_sample(dtshape: ia.DTShape, cfg: ia.Config = None, **kwargs):
    """Return random floats in the half-open interval [0.0, 1.0).

    Results are from the "continuous uniform" distribution.

    Parameters
    ----------
    dtshape : ia.DTShape
        The shape and data type of the array to be created.
    cfg : ia.Config
        The configuration for running the expression.
        If None (default), global defaults are used.
        In particular, `cfg.seed` and `cfg.random_gen` are honored
        in this context.
    kwargs : dict
        A dictionary for setting some or all of the fields in the ia.Config
        dataclass that should override the current configuration.
        In particular, `seed=` and `random_gen=` arguments are honored
        in this context.

    Returns
    -------
    ia.IArray
        The new array.

    See Also
    --------
    np.random.random_sample
    """
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.random_rand(cfg, dtshape)


def standard_normal(dtshape: ia.DTShape, cfg: ia.Config = None, **kwargs):
    """Draw samples from a standard Normal distribution (mean=0, stdev=1).

    The `dtshape`, `cfg` and `kwargs` parameters are the same than in ia.random.random_sample.

    Returns
    -------
    ia.IArray
        The new array.

    See Also
    --------
    ia.random.random_sample
    np.random.standard_normal
    """
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.random_randn(cfg, dtshape)


def beta(dtshape: ia.DTShape, alpha: float, beta: float, cfg: ia.Config = None, **kwargs):
    """Draw samples from a Beta distribution.

    The `dtshape`, `cfg` and `kwargs` parameters are the same than in ia.random.random_sample.

    Parameters
    ----------
    alpha : float
        Alpha, positive (>0).
    beta : float
        Beta, positive (>0).

    Returns
    -------
    ia.IArray
        The new array.

    See Also
    --------
    ia.random.random_sample
    np.random.beta
    """
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.random_beta(cfg, alpha, beta, dtshape)


def lognormal(
    dtshape: ia.DTShape, mean: float = 0.0, sigma: float = 1.0, cfg: ia.Config = None, **kwargs
):
    """Draw samples from a log-normal distribution.

    The `dtshape`, `cfg` and `kwargs` parameters are the same than in ia.random.random_sample.

    Parameters
    ----------
    mean : float or array_like of floats, optional
        Mean value of the underlying normal distribution. Default is 0.
    sigma : float or array_like of floats, optional
        Standard deviation of the underlying normal distribution. Must be
        non-negative. Default is 1.

    Returns
    -------
    ia.IArray
        The new array.

    See Also
    --------
    ia.random.random_sample
    np.random.lognormal
    """
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.random_lognormal(cfg, mean, sigma, dtshape)


def exponential(dtshape: ia.DTShape, scale: float = 1.0, cfg: ia.Config = None, **kwargs):
    """Draw samples from an exponential distribution.

    The `dtshape`, `cfg` and `kwargs` parameters are the same than in ia.random.random_sample.

    Parameters
    ----------
    scale : float
        The scale parameter, :math:`\\beta = 1/\\lambda`. Must be
        non-negative.

    Returns
    -------
    ia.IArray
        The new array.

    See Also
    --------
    ia.random.random_sample
    np.random.exponential
    """
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.random_exponential(cfg, scale, dtshape)


def uniform(
    dtshape: ia.DTShape, low: float = 0.0, high: float = 1.0, cfg: ia.Config = None, **kwargs
):
    """Draw samples from a uniform distribution.

    The `dtshape`, `cfg` and `kwargs` parameters are the same than in ia.random.random_sample.

    Parameters
    ----------
    low : float
        Lower boundary of the output interval.  All values generated will be
        greater than or equal to low.  The default value is 0.
    high : float
        Upper boundary of the output interval.  All values generated will be
        less than or equal to high.  The default value is 1.0.

    Returns
    -------
    ia.IArray
        The new array.

    See Also
    --------
    ia.random.random_sample
    np.random.uniform
    """
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.random_uniform(cfg, low, high, dtshape)


def normal(dtshape: ia.DTShape, loc: float, scale: float, cfg: ia.Config = None, **kwargs):
    """Draw random samples from a normal (Gaussian) distribution.

    The `dtshape`, `cfg` and `kwargs` parameters are the same than in ia.random.random_sample.

    Parameters
    ----------
    loc : float
        Mean ("centre") of the distribution.
    scale : float
        Standard deviation (spread or "width") of the distribution. Must be
        non-negative.

    Returns
    -------
    ia.IArray
        The new array.

    See Also
    --------
    ia.random.random_sample
    np.random.normal
    """
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.random_normal(cfg, loc, scale, dtshape)


def bernoulli(dtshape: ia.DTShape, p: float, cfg: ia.Config = None, **kwargs):
    """Draw samples from a Bernoulli distribution.

    The Bernoulli distribution is a special case of the binomial distribution where a
    single trial is conducted (so n would be 1 for such a binomial distribution).

    The `dtshape`, `cfg` and `kwargs` parameters are the same than in ia.random.random_sample.

    Parameters
    ----------
    p : float
        Parameter of the distribution, >= 0 and <=1.

    Returns
    -------
    ia.IArray
        The new array.

    See Also
    --------
    ia.random.random_sample
    ia.random.binomial
    """
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.random_bernoulli(cfg, p, dtshape)


def binomial(dtshape: ia.DTShape, n: float, p: float, cfg: ia.Config = None, **kwargs):
    """Draw samples from a binomial distribution.

    The `dtshape`, `cfg` and `kwargs` parameters are the same than in ia.random.random_sample.

    Parameters
    ----------
    n : int or array_like of ints
        Parameter of the distribution, >= 0. Floats are also accepted,
        but they will be truncated to integers.
    p : float
        Parameter of the distribution, >= 0 and <=1.

    Returns
    -------
    ia.IArray
        The new array.

    See Also
    --------
    ia.random.random_sample
    np.random.binomial
    """
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.random_binomial(cfg, n, p, dtshape)


def poisson(dtshape: ia.DTShape, lam: float, cfg: ia.Config = None, **kwargs):
    """Draw samples from a Poisson distribution.

    The `dtshape`, `cfg` and `kwargs` parameters are the same than in ia.random.random_sample.

    Parameters
    ----------
    lam : float
        Expectation of interval, must be >= 0.

    Returns
    -------
    ia.IArray
        The new array.

    See Also
    --------
    ia.random.random_sample
    np.random.poisson
    """
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.random_poisson(cfg, lam, dtshape)


def kstest(a: ia.IArray, b: ia.IArray, cfg: ia.Config = None, **kwargs):
    """Kolmogorov–Smirnov test of the equality of two distributions.

    This is mainly used for testing purposes.

    The `dtshape`, `cfg` and `kwargs` parameters are the same than in ia.random.random_sample.

    Parameters
    ----------
    a : ia.IArray
        First distribution.
    b : ia.IArray
        Second distribution.

    Returns
    -------
    bool
        Whether the two distributions are equal or not.

    See Also
    --------
    ia.random.random_sample
    np.random.poisson
    """
    with ia.config(cfg=cfg, **kwargs) as cfg:
        return ext.random_kstest(cfg, a, b)