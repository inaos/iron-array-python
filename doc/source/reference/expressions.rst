-----------
Expressions
-----------

ironArray has a strong support for expression evaluation. Things like sums, products, divisions and a pretty complete range of transcendental functions (e.g. exp, sin, asin, tanh…) have been implemented so as to guarantee an efficient evaluation in (large) arrays.

.. currentmodule:: iarray

Expr class
==========

.. autosummary::
   :toctree: autofiles/expressions/
   :nosignatures:

   Expr

Constructors
------------

.. autosummary::
   :toctree: autofiles/expressions/
   :nosignatures:

   expr_from_string
   expr_from_udf


Methods
-------

.. autosummary::
   :toctree: autofiles/expressions/
   :nosignatures:

   Expr.eval

