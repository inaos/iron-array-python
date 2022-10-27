.. _IArray:

----------------
IArray container
----------------

Multidimensional and compressed data container.

.. currentmodule:: iarray


Attributes
==========


.. autosummary::
   :toctree: autofiles/ndarray
   :nosignatures:

   IArray.blocks
   IArray.chunks
   IArray.cratio
   IArray.data
   IArray.dtype
   IArray.ndim
   IArray.shape
   IArray.info
   IArray.attrs
   IArray.is_view


Methods
=======

.. autosummary::
   :toctree: autofiles/ndarray
   :nosignatures:

   IArray.copy
   IArray.copyto
   IArray.transpose
   IArray.resize
   IArray.insert
   IArray.append
   IArray.delete
   IArray.split
   IArray.get_orthogonal_selection
   IArray.set_orthogonal_selection
   IArray.astype


Mathematical methods
--------------------

.. autosummary::
   :toctree: autofiles/ndarray/
   :nosignatures:

   IArray.sin
   IArray.cos
   IArray.tan
   IArray.arcsin
   IArray.arccos
   IArray.arctan
   IArray.arctan2
   IArray.asin
   IArray.acos
   IArray.atan
   IArray.atan2
   IArray.sinh
   IArray.cosh
   IArray.tanh
   IArray.floor
   IArray.ceil
   IArray.exp
   IArray.log
   IArray.log10
   IArray.sqrt
   IArray.power
   IArray.abs
   IArray.negative


.. seealso::

   :ref:`Mathematical Functions`


Reductions
----------

.. autosummary::
   :toctree: autofiles/ndarray/
   :nosignatures:

   IArray.max
   IArray.min
   IArray.sum
   IArray.prod
   IArray.mean
   IArray.var
   IArray.std
   IArray.median
   IArray.nanmax
   IArray.nanmin
   IArray.nansum
   IArray.nanprod
   IArray.nanmean
   IArray.nanvar
   IArray.nanstd
   IArray.nanmedian


.. seealso::

   :ref:`Reductions_doc`