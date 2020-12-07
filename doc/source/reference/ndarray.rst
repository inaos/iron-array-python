-------------------
N-Dimensional Array
-------------------

.. currentmodule:: iarray

Dealing with array shapes and data types
========================================

.. autosummary::
   :toctree: ndarray

   DTShape


Attributes
==========


.. autosummary::
   :toctree: ndarray

   IArray.blockshape
   IArray.chunkshape
   IArray.cratio
   IArray.data
   IArray.dtshape
   IArray.dtype
   IArray.is_plainbuffer
   IArray.ndim
   IArray.shape


Methods
=======

.. autosummary::
   :toctree: ndarray

   IArray.copy
   IArray.copyto
   IArray.transpose


Utilities
=========

.. autosummary::
   :toctree: ndarray
   :nosignatures:

   load
   save
   cmp_arrays
   iarray2numpy
   numpy2iarray
   get_ncores
   partition_advice
