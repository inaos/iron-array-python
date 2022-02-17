iarray.IArray.vlmeta
====================

.. currentmodule:: iarray

.. autoattribute:: IArray.vlmeta

Instance to a vlmeta object to manage the variable length metalayers of an `IArray`.
This class behaves very similarly to a dictionary, and variable length
metalayers can be appended in the usual way::

    iarr.vlmeta['vlmeta1'] = 'something'

And can be retrieved similarly::

    value = iarr.vlmeta['vlmeta1']

Once added, a vlmeta can be deleted with::

    del iarr.vlmeta['vlmeta1']

This class also honors the `__contains__` and `__len__` special
functions.  Moreover, a `getall()` method returns all the
variable length metalayers as a dictionary.