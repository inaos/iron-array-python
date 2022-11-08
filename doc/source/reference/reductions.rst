.. _Reductions_doc:

----------
Reductions
----------

.. currentmodule:: iarray

Reductions behave differently than mathematical functions in the sense that they always return an :ref:`IArray`,
that is, the evaluation is done immediately and not in a lazy way as in mathematical functions. This is why
the functions and methods involving reductions do return an :ref:`IArray` and not a Lazy :py:class:`iarray.Expr`.

.. autosummary::
   :toctree: autofiles/reductions/
   :nosignatures:

   all
   any
   max
   min
   sum
   prod
   mean
   var
   std
   median
   nanmax
   nanmin
   nansum
   nanprod
   nanmean
   nanvar
   nanstd
   nanmedian
