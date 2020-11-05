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
from dataclasses import dataclass, field, fields, replace
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
    # Config params
    _cparams = None
    codec: ia.Codecs = ia.Codecs.LZ4
    clevel: int = 5
    use_dict: bool = False
    filters: List[ia.Filters] = field(default_factory=default_filters)
    nthreads: int = 0
    fp_mantissa_bits: int = 0
    eval_method: int = ia.Eval.AUTO
    seed: int = None
    # Storage
    _storage = None
    chunkshape: Sequence = None
    blockshape: Sequence = None
    filename: str = None
    enforce_frame: bool = False
    plainbuffer: bool = False

    def __post_init__(self):
        # Initialize cparams and storage with its getters and setters
        self.cparams = self.cparams

    # Accessors only meant to serve as default_factory
    def _codec(self):
        return self.codec

    def _clevel(self):
        return self.clevel

    def _use_dict(self):
        return self.use_dict

    def _filters(self):
        return self.filters

    def _nthreads(self):
        return self.nthreads

    def _fp_mantissa_bits(self):
        return self.fp_mantissa_bits

    def _eval_method(self):
        return self.eval_method

    def _seed(self):
        return self.seed

    @property
    def cparams(self):
        if self._cparams is None:
            # Bootstrap the defaults
            return DefaultConfigParams(
                codec=self.codec,
                clevel=self.clevel,
                use_dict=self.use_dict,
                filters=self.filters,
                nthreads=self.nthreads,
                fp_mantissa_bits=self.fp_mantissa_bits,
                storage=self.storage,
                eval_method=self.eval_method,
                seed=self.seed,
            )
        return self._cparams

    @cparams.setter
    def cparams(self, value):
        if not hasattr(value, "codec"):
            raise ValueError(f"You need to use a `ConfigParams` instance")
        self.codec = value.codec
        self.clevel = value.clevel
        self.use_dict = value.use_dict
        self.filters = value.filters
        self.nthreads = value.nthreads
        self.fp_mantissa_bits = value.fp_mantissa_bits
        self.eval_method = value.eval_method
        self.seed = value.seed
        self._storage = value.storage
        self._cparams = value
        if self._storage is not None:
            self.set_storage(self._storage)

    def _chunkshape(self):
        return self.chunkshape

    def _blockshape(self):
        return self.blockshape

    def _filename(self):
        return self.filename

    def _enforce_frame(self):
        return self.enforce_frame

    def _plainbuffer(self):
        return self.plainbuffer

    @property
    def storage(self):
        if self._storage is None:
            # Bootstrap the defaults
            return DefaultStorage(
                chunkshape=self.chunkshape,
                blockshape=self.blockshape,
                filename=self.filename,
                enforce_frame=self.enforce_frame,
                plainbuffer=self.plainbuffer,
            )
        return self._storage

    def set_storage(self, value):
        if not hasattr(value, "chunkshape"):
            raise ValueError(f"You need to use a `Storage` instance")
        self.chunkshape = value.chunkshape
        self.blockshape = value.blockshape
        self.filename = value.filename
        self.enforce_frame = value.enforce_frame
        self.plainbuffer = value.plainbuffer
        self._storage = value


# Global variable where the defaults for config params are stored
defaults = Defaults()


@dataclass
class Storage:
    chunkshape: Union[Sequence, None] = field(default_factory=defaults._chunkshape)
    blockshape: Union[Sequence, None] = field(default_factory=defaults._blockshape)
    filename: str = field(default_factory=defaults._filename)
    enforce_frame: bool = field(default_factory=defaults._enforce_frame)
    plainbuffer: bool = field(default_factory=defaults._plainbuffer)

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
    """
    Dataclass for hosting the different ironArray parameters.
    """

    codec: ia.Codecs = field(default_factory=defaults._codec)
    clevel: int = field(default_factory=defaults._clevel)
    use_dict: bool = field(default_factory=defaults._use_dict)
    filters: List[ia.Filters] = field(default_factory=defaults._filters)
    nthreads: int = field(default_factory=defaults._nthreads)
    fp_mantissa_bits: int = field(default_factory=defaults._fp_mantissa_bits)
    storage: Storage = None  # delayed initialization
    eval_method: int = field(default_factory=defaults._eval_method)
    seed: int = field(default_factory=defaults._seed)

    # These belong to Storage, but we accept them in top level too
    chunkshape: Union[Sequence, None] = field(default_factory=defaults._chunkshape)
    blockshape: Union[Sequence, None] = field(default_factory=defaults._blockshape)
    filename: str = field(default_factory=defaults._filename)
    enforce_frame: bool = field(default_factory=defaults._enforce_frame)
    plainbuffer: bool = field(default_factory=defaults._plainbuffer)

    def __post_init__(self):
        global RANDOM_SEED
        if self.nthreads == 0:  # trigger automatic core detection
            # As a general rule, it is useful to get just the physical cores.
            # The rational is that logical cores share the L1 and L2 caches, and
            # usually it is better to let 1 single thread to use L1 and L2
            # simultaneously.
            self.nthreads = get_ncores(0)
        else:
            # If number of threads is specified, make sure that we are not exceeding
            # the number of physical cores in the system.
            self.nthreads = get_ncores(self.nthreads)
        if self.nthreads < 1:
            self.nthreads = 1

        # Increase the random seed each time so as to prevent re-using them
        if self.seed is None:
            if RANDOM_SEED >= 2 ** 32 - 1:
                # In case we run out of values in uint32_t ints, reset to 0
                RANDOM_SEED = 0
            RANDOM_SEED += 1
            self.seed = RANDOM_SEED
        if self.storage is None:
            self.storage = Storage(
                chunkshape=self.chunkshape,
                blockshape=self.blockshape,
                filename=self.filename,
                enforce_frame=self.enforce_frame,
                plainbuffer=self.plainbuffer,
            )

        # Initialize the Cython counterpart
        super().__init__(
            self.codec,
            self.clevel,
            self.use_dict,
            self.filters,
            self.nthreads,
            self.fp_mantissa_bits,
            self.eval_method,
        )

    def replace(self, **kwargs):
        cfg_ = replace(self, **kwargs)
        store_args = dict(
            (field.name, kwargs[field.name]) for field in fields(Storage) if field.name in kwargs
        )
        cfg_.storage = replace(cfg_.storage, **store_args)
        return cfg_


global_defaults = None


def set_config(cfg: ConfigParams = None, dtshape=None, **kwargs):
    """Set the global defaults for iarray operations.

    `cfg` is a `ConfigParams` instance.  If None, sensible defaults will apply.
    Use `get_config()` before anything so as to see which those defaults are.

    `dtshape` is a `DTShape` instance.  This is not part of `ConfigParams` as such,
    but if passed, it will be used so as to compute sensible defaults for `chunkshape`
    and `blockshape`.

    `**kwargs` is a dictionary for setting some of the fields in `ConfigParams`
    dataclass different than defaults.

    Returns the new global configuration.
    """
    global global_defaults
    if cfg is None:
        if global_defaults is None:
            cfg = ConfigParams()
        else:
            cfg = global_defaults

    if kwargs != {}:
        cfg = cfg.replace(**kwargs)
    if dtshape is not None:
        cfg.storage.get_shape_advice(dtshape)

    global_defaults = cfg
    # Set the defaults for ConfigParams() constructor and other nested confs (Storage...)
    defaults.cparams = cfg

    return global_defaults


# Initialize the configuration
set_config()


def get_config():
    """Get the global defaults for iarray operations.

    Returns the existing global configuration.
    """
    return global_defaults


@contextmanager
def config(cfg: ConfigParams = None, dtshape=None, **kwargs):
    """Execute a context with some configuration parameters.

    All parameters are and work the same than in `ia.set_config`.
    The only difference is that this does not set global defaults.
    """
    if cfg is None:
        cfg = ConfigParams()
    cfg_ = cfg.replace(**kwargs)
    if dtshape is not None:
        cfg_.storage.get_shape_advice(dtshape)

    yield cfg_


if __name__ == "__main__":
    cfg_ = get_config()
    print("Defaults:", cfg_)
    assert cfg_.storage.enforce_frame == False

    set_config(storage=Storage(enforce_frame=True))
    cfg = get_config()
    print("1st form:", cfg)
    assert cfg.storage.enforce_frame == True

    set_config(enforce_frame=False)
    cfg = get_config()
    print("2nd form:", cfg)
    assert cfg.storage.enforce_frame == False

    set_config(ConfigParams(clevel=1))
    cfg = get_config()
    print("3rd form:", cfg)
    assert cfg.clevel == 1

    with config(clevel=0, enforce_frame=True) as cfg_new:
        print("Context form:", cfg_new)
        assert cfg_new.storage.enforce_frame == True

    cfg = ia.ConfigParams(codec=ia.Codecs.BLOSCLZ)
    cfg2 = ia.set_config(cfg=cfg, codec=ia.Codecs.LIZARD)
    print("Standalone config:", cfg)
    print("Global config", cfg2)

    cfg = ia.set_config(cfg_)
    print("Defaults config:", cfg)
