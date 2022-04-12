-----------
Expressions
-----------

ironArray has a strong support for expression evaluation. Things like sums,
products, divisions and a pretty complete range of transcendental functions
(e.g. exp, sin, asin, tanh…) have been implemented so as to guarantee an
efficient evaluation in (large) arrays.

Expressions can be built either from small one liners (either in string format
or as regular Python expressions, see tutorials section for details), or from
User Defined Functions which are described later in section :ref:`UDFs`. There are
the next constructors for creating an expression:

.. currentmodule:: iarray

Constructors
------------

.. autosummary::
   :toctree: autofiles/expressions/
   :nosignatures:

   expr_from_string
   expr_from_udf
   expr_get_operands


Expressions are implemented in the :py:class:`iarray.Expr` class.  Once built,
they can be evaluated via the :func:`iarray.Expr.eval` method.

Expr class
==========

.. autosummary::
   :toctree: autofiles/expressions/
   :nosignatures:

   Expr

Methods
-------

.. autosummary::
   :toctree: autofiles/expressions/
   :nosignatures:

   Expr.eval
