-----------
Expressions
-----------

ironArray has a strong support for expression evaluation. Things like sums,
products, divisions and a pretty complete range of transcendental functions
(e.g. exp, sin, asin, tanh…) have been implemented so as to guarantee an
efficient evaluation in (large) arrays.

Expressions can be built either from small one liners (either in string format
or as regular Python expressions, see tutorials section for details), or from
User Defined Functions which are described later in section :ref:`UDFs`.

Expressions are implemented in the :py:class:`iarray.Expr` class.  Once built,
they can be evaluated via the :func:`iarray.Expr.eval` method.

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
