iarray.IArray.attrs
===================

.. currentmodule:: iarray

.. autoattribute:: IArray.attrs

Instance to a `Blosc2 <https://c-blosc2.readthedocs.io/en/latest/ >`_ Attrs class to
manage the attributes of an `IArray`.
This class behaves very similarly to a dictionary, and attributes
can be appended in the usual way::

    iarr.attrs['attr1'] = 'something'

And can be retrieved similarly::

    value = iarr.attrs['attr1']

Once added, an attribute can be deleted with::

    del iarr.attrs['attr1']

This class also honors the `__contains__` special
function.
