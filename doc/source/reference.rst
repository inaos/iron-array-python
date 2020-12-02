-----------------
Library Reference
-----------------

.. currentmodule:: iarray


Main classes
=============

.. autoclass:: iarray.IArray

.. autoclass:: iarray.DTShape
   :members: shape, dtype


Constructors
============

.. autofunction:: iarray.empty
.. autofunction:: iarray.arange
.. autofunction:: iarray.linspace
.. autofunction:: iarray.zeros
.. autofunction:: iarray.ones
.. autofunction:: iarray.full


Random Constructors
===================

.. autofunction:: iarray.random.random_sample
.. autofunction:: iarray.random.standard_normal
.. autofunction:: iarray.random.beta
.. autofunction:: iarray.random.lognormal
.. autofunction:: iarray.random.exponential
.. autofunction:: iarray.random.uniform
.. autofunction:: iarray.random.normal
.. autofunction:: iarray.random.bernoulli
.. autofunction:: iarray.random.binomial
.. autofunction:: iarray.random.poisson
.. autofunction:: iarray.random.kstest


Universal Functions
===================

.. autofunction:: iarray.abs
.. autofunction:: iarray.arccos
.. autofunction:: iarray.arcsin
.. autofunction:: iarray.arctan
.. autofunction:: iarray.arctan2
.. autofunction:: iarray.ceil
.. autofunction:: iarray.cos
.. autofunction:: iarray.cosh
.. autofunction:: iarray.exp
.. autofunction:: iarray.floor
.. autofunction:: iarray.log
.. autofunction:: iarray.log10
.. autofunction:: iarray.negative
.. autofunction:: iarray.power
.. autofunction:: iarray.sin
.. autofunction:: iarray.sinh
.. autofunction:: iarray.sqrt
.. autofunction:: iarray.tan
.. autofunction:: iarray.tanh


Linear Algebra Operations
=========================

.. autofunction:: iarray.matmul
.. autofunction:: iarray.transpose

Reductions
==========

.. autofunction:: iarray.max
.. autofunction:: iarray.min
.. autofunction:: iarray.sum
.. autofunction:: iarray.prod
.. autofunction:: iarray.mean

Expressions
===========

.. autoclass:: iarray.Expr
    :members: eval
.. autoclass:: iarray.LazyExpr
    :members: eval

.. autofunction:: iarray.expr_from_string
.. autofunction:: iarray.expr_from_udf


Set/get Configuration Parameters
================================

.. autoclass:: iarray.Config
.. autoclass:: iarray.Storage

.. autofunction:: iarray.set_config
.. autofunction:: iarray.get_config
.. autofunction:: iarray.config
.. autofunction:: iarray.reset_config_defaults


Utilities
=========

.. autofunction:: iarray.load
.. autofunction:: iarray.save
.. autofunction:: iarray.cmp_arrays
.. autofunction:: iarray.iarray2numpy
.. autofunction:: iarray.numpy2iarray
.. autofunction:: iarray.get_ncores
.. autofunction:: iarray.partition_advice


Enumerated Classes
==================

.. autoclass:: iarray.Codecs
.. autoclass:: iarray.Filters
.. autoclass:: iarray.Eval
.. autoclass:: iarray.RandomGen


Global Variables
================

.. py:attribute:: iarray.__version__
