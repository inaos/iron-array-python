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

import iarray as ia
from iarray import iarray_ext as ext
from dataclasses import dataclass, field
from typing import List, Sequence
import warnings

# Global variable for random seed
RANDOM_SEED = 0


def get_ncores(max_ncores=0):
    """Return the number of logical cores in the system.

    This number is capped at `max_ncores`.  When `max_ncores` is 0,
    there is no cap at all.
    """
    ncores = ext.get_ncores(max_ncores)
    if ncores < 0:
        warnings.warn(
            "Error getting the number of cores in this system (please report this)."
            "  Falling back to 1.",
            UserWarning,
        )
        return 1
    return ncores


def partition_advice(
    dtshape, min_chunksize=0, max_chunksize=0, min_blocksize=0, max_blocksize=0, config=None
):
    """Provide advice for the chunk and block shapes for a certain `dtshape`.

    `min_` and `max_` params contain minimum and maximum values for chunksize and blocksize.
    If `min_` or `max_` are 0, they default to sensible values (fractions of CPU caches).

    `config` is an `Config` instance, and if not passed, a default configuration is used.

    If success, the tuple (chunkshape, blockshape) containing the advice is returned.
    In case of error, a (None, None) tuple is returned and a warning is issued.
    """
    if config is None:
        config = ConfigParams()
    chunkshape, blockshape = ext.partition_advice(
        dtshape, min_chunksize, max_chunksize, min_blocksize, max_blocksize, config
    )
    if chunkshape is None:
        warnings.warn(
            "Error in providing partition advice (please report this)."
            "  Please do not trust on the chunkshape and blockshape in `storage`!",
            UserWarning,
        )
    return chunkshape, blockshape


@dataclass
class Storage:
    chunkshape: Sequence = None
    blockshape: Sequence = None
    filename: str = None
    enforce_frame: bool = False
    plainbuffer: bool = False

    def __post_init__(self):
        self.enforce_frame = True if self.filename else self.enforce_frame
        if self.plainbuffer:
            if self.chunkshape is not None or self.blockshape is not None:
                raise ValueError("plainbuffer array does not support a chunkshape or blockshape")

    def get_shape_advice(self, dtshape):
        if self.plainbuffer:
            return
        chunkshape, blockshape = self.chunkshape, self.blockshape
        if chunkshape is not None and blockshape is not None:
            return
        if chunkshape is None and blockshape is None:
            chunkshape_, blockshape_ = partition_advice(dtshape)
            self.chunkshape = chunkshape_
            self.blockshape = blockshape_
            return
        else:
            raise ValueError(
                "You can either specify both chunkshape and blockshape or none of them."
            )


def default_storage():
    return Storage()


def default_filters():
    return [ia.Filters.SHUFFLE]


@dataclass
class ConfigParams(ext.ConfigParams):
    codec: int = ia.Codecs.LZ4
    clevel: int = 5
    use_dict: bool = False
    filters: List[int] = field(default_factory=default_filters)
    nthreads: int = 0
    fp_mantissa_bits: int = 0
    storage: Storage = field(default_factory=default_storage)
    eval_method: int = ia.Eval.AUTO
    seed: int = 0

    def __post_init__(self):
        self.nthreads = get_ncores(self.nthreads)
        if self.storage is None:
            self.storage = Storage()

        super().__init__(
            self.codec,
            self.clevel,
            self.use_dict,
            self.filters,
            self.nthreads,
            self.fp_mantissa_bits,
            self.eval_method,
        )
