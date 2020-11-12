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


def rand(dtshape, cfg=None, **kwargs):
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.random_rand(cfg, dtshape)


def randn(dtshape, cfg=None, **kwargs):
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.random_randn(cfg, dtshape)


def beta(dtshape, alpha, beta, cfg=None, **kwargs):
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.random_beta(cfg, alpha, beta, dtshape)


def lognormal(dtshape, mu, sigma, cfg=None, **kwargs):
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.random_lognormal(cfg, mu, sigma, dtshape)


def exponential(dtshape, beta, cfg=None, **kwargs):
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.random_exponential(cfg, beta, dtshape)


def uniform(dtshape, a, b, cfg=None, **kwargs):
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.random_uniform(cfg, a, b, dtshape)


def normal(dtshape, mu, sigma, cfg=None, **kwargs):
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.random_normal(cfg, mu, sigma, dtshape)


def bernoulli(dtshape, p, cfg=None, **kwargs):
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.random_bernoulli(cfg, p, dtshape)


def binomial(dtshape, m, p, cfg=None, **kwargs):
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.random_binomial(cfg, m, p, dtshape)


def poisson(dtshape, lamb, cfg=None, **kwargs):
    with ia.config(dtshape=dtshape, cfg=cfg, **kwargs) as cfg:
        return ext.random_poisson(cfg, lamb, dtshape)


def kstest(a, b, cfg=None, **kwargs):
    with ia.config(cfg=cfg, **kwargs) as cfg:
        return ext.random_kstest(cfg, a, b)
