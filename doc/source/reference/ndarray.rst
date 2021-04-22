-------------------
N-Dimensional Array
-------------------

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
   IArray.is_plainbuffer
   IArray.ndim
   IArray.shape
   IArray.info


Methods
=======

.. autosummary::
   :toctree: autofiles/ndarray
   :nosignatures:

   IArray.copy
   IArray.copyto
   IArray.transpose


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

Utilities
=========

.. autosummary::
   :toctree: autofiles/ndarray
   :nosignatures:

   load
   open
   save
   cmp_arrays
   iarray2numpy
   numpy2iarray
   get_ncores
   partition_advice
