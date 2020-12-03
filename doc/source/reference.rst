-----------------
Library Reference
-----------------

.. currentmodule:: iarray


Main classes
=============

.. autosummary::
   :toctree: reference/
   :nosignatures:


   IArray
   DTShape

Constructors
============

.. autosummary::
   :toctree: reference/
   :nosignatures:


   empty
   arange
   linspace
   zeros
   ones
   full


Random Constructors
===================
.. autosummary::
   :toctree: reference/
   :nosignatures:


   random.random_sample
   random.standard_normal
   random.beta
   random.lognormal
   random.exponential
   random.uniform
   random.normal
   random.bernoulli
   random.binomial
   random.poisson
   random.kstest


Universal Functions
===================
.. autosummary::
   :toctree: reference/
   :nosignatures:


   abs
   arccos
   arcsin
   arctan
   arctan2
   ceil
   cos
   cosh
   exp
   floor
   log
   log10
   negative
   power
   sin
   sinh
   sqrt
   tan
   tanh


Linear Algebra Operations
=========================

.. autosummary::
   :toctree: reference/
   :nosignatures:


   matmul
   transpose

Reductions
==========

.. autosummary::
   :toctree: reference/
   :nosignatures:

   max
   min
   sum
   prod
   mean

Expressions
===========

.. autosummary::
   :toctree: reference/
   :nosignatures:

   Expr
   LazyExpr
   expr_from_string
   expr_from_udf


Set/get Configuration Parameters
================================

.. autosummary::
   :toctree: reference/
   :nosignatures:

   Config
   Storage
   set_config
   get_config
   config
   reset_config_defaults


Utilities
=========

.. autosummary::
   :toctree: reference/
   :nosignatures:

   load
   save
   cmp_arrays
   iarray2numpy
   numpy2iarray
   get_ncores
   partition_advice


Enumerated Classes
==================

.. autosummary::
   :toctree: reference/
   :nosignatures:

   Codecs
   Filters
   Eval
   RandomGen


Global Variables
================

.. py:attribute:: iarray.__version__
