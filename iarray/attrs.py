###########################################################################################
# Copyright INAOS GmbH, Thalwil, 2018.
# Copyright Francesc Alted, 2018.
#
# All rights reserved.
#
# This software is the confidential and proprietary information of INAOS GmbH
# and Francesc Alted ("Confidential Information"). You shall not disclose such Confidential
# Information and shall use it only in accordance with the terms of the license agreement.
###########################################################################################


from iarray import iarray_ext as ext
import iarray as ia
from collections.abc import MutableMapping
from msgpack import packb, unpackb


class Attributes(MutableMapping):

    def __init__(self, a: ia.IArray):
        self.iarr = a

    def __getitem__(self, name):
        """Get the content of an attr.

        Parameters
        ----------
        name : str or byte string
            The name of the attr to return.

        Returns
        -------
        object :
            The unpacked content of the attr.
        """
        packed_content = ext.attr_getitem(self.iarr, name)
        return unpackb(packed_content)

    def __setitem__(self, name, content):
        """Add or update an attr.

        Its content will be packed with msgpack and unpacked when getting it.

        Parameters
        ----------
        name : str or byte string
            The name of the attribute.
        content : object
            The content of the attr. The object must be an object that can be serialized by `msgpack.packb`.
            The getitem counterpart will `msgpack.unpackb` it automatically.

        Returns
        -------
        None

        """
        packed_content = packb(content)
        ext.attr_setitem(self.iarr, name, packed_content)

    def __delitem__(self, name):
        """Delete the attr given by :paramref:`name`.

        Parameters
        ----------
        name : str or byte string
            The name of the attr.
        """
        ext.attr_delitem(self.iarr, name)

    def __iter__(self):
        keys = ext.attr_get_names(self.iarr)
        for name in keys:
            yield name

    def __len__(self):
        return ext.attr_len(self.iarr)

