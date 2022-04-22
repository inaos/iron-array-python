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


Expressions are implemented in the :py:class:`iarray.Expr` class.  Once built,
they can be evaluated via the :func:`iarray.Expr.eval` method.

Expressions can call UDF functions registered via the
:py:class:`iarray.UdfRegistry` class.

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

UdfRegistry
===========

This class is meant to register UDF functions in libraries.
As there is only a global register, the user must use its global
`udf_registry` instance in the iarray package.

Note that, since the inclusion of the new `lib=` param in the `udf.scalar`
decorator, it is not necessary to use this explicitly for registering
anymore.  See `udf_expr.py` example for more info on how to use the
register mechanism.

.. autosummary::
   :toctree: autofiles/expressions/
   :nosignatures:

   UdfRegistry

Methods
-------

.. autosummary::
   :toctree: autofiles/expressions/
   :nosignatures:

   UdfRegistry.__getitem__
   UdfRegistry.__setitem__
   UdfRegistry.__delitem__
   UdfRegistry.iter_funcs
   UdfRegistry.iter_all_func_names
   UdfRegistry.get_func_addr
   UdfRegistry.clear

Utils
-----

.. autosummary::
   :toctree: autofiles/expressions/
   :nosignatures:

   expr_get_operands
   expr_get_ops_funcs
