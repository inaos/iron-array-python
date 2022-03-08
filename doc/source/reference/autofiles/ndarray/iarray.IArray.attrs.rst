iarray.IArray.attrs
===================

.. currentmodule:: iarray

.. autoattribute:: IArray.attrs

Instance to a `Blosc2 <https://c-blosc2.readthedocs.io/en/latest/ >`_ variable length metalayer object to
manage the variable length metalayers of an `IArray`.
This class behaves very similarly to a dictionary, and variable length
metalayers can be appended in the usual way::

    iarr.attrs['vlmeta1'] = 'something'

And can be retrieved similarly::

    value = iarr.attrs['vlmeta1']

Once added, a vlmeta can be deleted with::

    del iarr.attrs['vlmeta1']

This class also honors the `__contains__` special
function.
