iarray.IArray.attrs
===================

.. currentmodule:: iarray

.. autoattribute:: IArray.attrs

Instance to a `Blosc2 <https://c-blosc2.readthedocs.io/en/latest/>`_ `Attributes` class to
manage the attributes of an `IArray`.
This class inherites from
`MutableMapping <https://docs.python.org/3/library/collections.abc.html#collections.abc.MutableMapping>`_
so every method in this class is available.

This class behaves very similarly to a dictionary, and attributes can be appended in the usual way::

     iarr.attrs['attr1'] = 'something'

 And can be retrieved similarly::

     value = iarr.attrs['attr1']

 Once added, an attribute can be deleted with::

     del iarr.attrs['attr1']

Methods
-------

.. autosummary::
   :toctree: attrs
   :nosignatures:

    Attributes.__getitem__
    Attributes.__setitem__
    Attributes.__delitem__
    Attributes.__iter__
    Attributes.__len__
    Attributes.__contains__
    Attributes.popitem
    Attributes.pop
    Attributes.values
    Attributes.keys
    Attributes.items
    Attributes.clear
    Attributes.update
    Attributes.setdefault
    Attributes.get
    Attributes.__eq__
    Attributes.__ne__
