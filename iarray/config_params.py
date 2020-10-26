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
from typing import List, Sequence, Any, Union
import warnings
from contextlib import contextmanager


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
    dtshape, min_chunksize=0, max_chunksize=0, min_blocksize=0, max_blocksize=0, cparams=None
):
    """Provide advice for the chunk and block shapes for a certain `dtshape`.

    `min_` and `max_` params contain minimum and maximum values for chunksize and blocksize.
    If `min_` or `max_` are 0, they default to sensible values (fractions of CPU caches).

    `config` is a `ConfigParams` instance, and if not passed, a default configuration is used.

    If success, the tuple (chunkshape, blockshape) containing the advice is returned.
    In case of error, a (None, None) tuple is returned and a warning is issued.
    """
    if cparams is None:
        cparams = ConfigParams()
    chunkshape, blockshape = ext.partition_advice(
        dtshape, min_chunksize, max_chunksize, min_blocksize, max_blocksize, cparams
    )
    if chunkshape is None:
        warnings.warn(
            "Error in providing partition advice (please report this)."
            "  Please do not trust on the chunkshape and blockshape in `storage`!",
            UserWarning,
        )
    return chunkshape, blockshape


@dataclass
class DefaultConfigParams:
    codec: Any
    clevel: Any
    use_dict: Any
    filters: Any
    nthreads: Any
    fp_mantissa_bits: Any
    storage: Any
    eval_method: Any
    seed: Any


@dataclass
class DefaultStorage:
    chunkshape: Any
    blockshape: Any
    filename: Any
    enforce_frame: Any
    plainbuffer: Any


def default_filters():
    return [ia.Filters.SHUFFLE]


@dataclass
class Defaults(object):
    """Class to set and get global default values."""

    # Config params
    _cparams = None
    _codec: ia.Codecs = ia.Codecs.LZ4
    _clevel: int = 5
    _use_dict: bool = False
    _filters: List[ia.Filters] = field(default_factory=default_filters)
    _nthreads: int = 0
    _fp_mantissa_bits: int = 0
    _eval_method: int = ia.Eval.AUTO
    _seed: int = None
    # Storage
    _storage = None
    _chunkshape: Sequence = None
    _blockshape: Sequence = None
    _filename: str = None
    _enforce_frame: bool = False
    _plainbuffer: bool = False

    def __post_init__(self):
        # Initialize cparams and storage with its getters and setters
        self.cparams = self.cparams

    # Accessors only meant to serve as default_factory
    def codec(self):
        return self._codec

    def clevel(self):
        return self._clevel

    def use_dict(self):
        return self._use_dict

    def filters(self):
        return self._filters

    def nthreads(self):
        return self._nthreads

    def fp_mantissa_bits(self):
        return self._fp_mantissa_bits

    def eval_method(self):
        return self._eval_method

    def seed(self):
        return self._seed

    @property
    def cparams(self):
        if self._cparams is None:
            # Bootstrap the defaults
            return DefaultConfigParams(
                codec=self._codec,
                clevel=self._clevel,
                use_dict=self._use_dict,
                filters=self._filters,
                nthreads=self._nthreads,
                fp_mantissa_bits=self._fp_mantissa_bits,
                storage=self._storage,
                eval_method=self._eval_method,
                seed=self._seed,
            )
        return self._cparams

    @cparams.setter
    def cparams(self, value):
        if not hasattr(value, "codec"):
            raise ValueError(f"You need to use a `ConfigParams` instance")
        self._codec = value.codec
        self._clevel = value.clevel
        self._use_dict = value.use_dict
        self._filters = value.filters
        self._nthreads = value.nthreads
        self._fp_mantissa_bits = value.fp_mantissa_bits
        self._storage = value.storage
        if self._storage is not None:
            self.set_storage(self._storage)
        self._eval_method = value.eval_method
        self._seed = value.seed
        self._cparams = value

    def chunkshape(self):
        return self._chunkshape

    def blockshape(self):
        return self._blockshape

    def filename(self):
        return self._filename

    def enforce_frame(self):
        return self._enforce_frame

    def plainbuffer(self):
        return self._plainbuffer

    @property
    def storage(self):
        if self._storage is None:
            # Bootstrap the defaults
            return DefaultStorage(
                chunkshape=self._chunkshape,
                blockshape=self._blockshape,
                filename=self._filename,
                enforce_frame=self._enforce_frame,
                plainbuffer=self._plainbuffer,
            )
        return self._storage

    def set_storage(self, value):
        if not hasattr(value, "chunkshape"):
            raise ValueError(f"You need to use a `Storage` instance")
        self._chunkshape = value.chunkshape
        self._blockshape = value.blockshape
        self._filename = value.filename
        self._enforce_frame = value.enforce_frame
        self._plainbuffer = value.plainbuffer
        self._storage = value


defaults = Defaults()


def set_config(**kwargs):
    defaults.cparams = ConfigParams(**kwargs)


def get_config():
    return defaults.cparams


@dataclass
class Storage:
    chunkshape: Union[Sequence, None] = field(default_factory=defaults.chunkshape)
    blockshape: Union[Sequence, None] = field(default_factory=defaults.blockshape)
    filename: str = field(default_factory=defaults.filename)
    enforce_frame: bool = field(default_factory=defaults.enforce_frame)
    plainbuffer: bool = field(default_factory=defaults.plainbuffer)

    def __post_init__(self):
        self.enforce_frame = True if self.filename else self.enforce_frame
        if self.plainbuffer:
            if self.chunkshape is not None or self.blockshape is not None:
                raise ValueError(
                    "plainbuffer array does not support neither a chunkshape nor blockshape"
                )

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


@dataclass
class ConfigParams(ext.ConfigParams):
    codec: ia.Codecs = field(default_factory=defaults.codec)
    clevel: int = field(default_factory=defaults.clevel)
    use_dict: bool = field(default_factory=defaults.use_dict)
    filters: List[ia.Filters] = field(default_factory=defaults.filters)
    nthreads: int = field(default_factory=defaults.nthreads)
    fp_mantissa_bits: int = field(default_factory=defaults.fp_mantissa_bits)
    storage: Storage = None  # delayed initialization
    eval_method: int = field(default_factory=defaults.eval_method)
    seed: int = field(default_factory=defaults.seed)

    def __post_init__(self):
        global RANDOM_SEED
        self.nthreads = get_ncores(self.nthreads)
        # Increase the random seed each time so as to prevent re-using them
        if self.seed is None:
            if RANDOM_SEED >= 2 ** 32 - 1:
                # In case we run out of values in uint32_t ints, reset to 0
                RANDOM_SEED = 0
            RANDOM_SEED += 1
            self.seed = RANDOM_SEED
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


# Initialize the configuration
set_config()


@contextmanager
def config(dtshape=None, **kwargs):
    """Execute a context with some defaults"""
    cparams_orig = defaults.cparams
    defaults.cparams = ConfigParams(**kwargs)
    if dtshape is not None:
        defaults.cparams.storage.get_shape_advice(dtshape)

    yield defaults.cparams

    defaults.cparams = cparams_orig


if __name__ == "__main__":
    cfg = get_config()
    print(cfg)
    set_config(storage=Storage(enforce_frame=True))
    cfg = get_config()
    print(cfg)

    with config(clevel=0, storage=Storage(plainbuffer=True)) as cfg:
        print(cfg)
