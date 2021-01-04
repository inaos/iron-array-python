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
    """Get the number of logical cores in the system.

    Parameters
    ----------
    max_ncores : int
        If > 0, the returned number is capped at this value.
        If == 0 (default), the actual number of logical cores in the system is returned.

    Returns
    -------
    int
        The (capped) number of logical cores.
        In case of error, a 1 is returned and a warning is issued.
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
    dtshape, min_chunksize=0, max_chunksize=0, min_blocksize=0, max_blocksize=0, cfg=None
):
    """Provide advice for the chunk and block shapes for a certain `dtshape`.

    Parameters
    ----------
    dtshape : ia.DTShape
        The shape and dtype of the array.
    min_chunksize : int
        Minimum value for chunksize (in bytes).  If 0 (default), a sensible value is chosen.
    max_chunksize : int
        Maximum value for chunksize (in bytes).  If 0 (default), a sensible value is chosen.
    min_bloksize : int
        Minimum value for blocksize (in bytes).  If 0 (default), a sensible value is chosen.
    max_bloksize : int
        Maximum value for blocksize (in bytes).  If 0 (default), a sensible value is chosen.
    cfg : ia.Config
        A configuration.  If None, the global configuration is used.

    Returns
    -------
    tuple
        If success, a (chunkshape, blockshape) containing the advice is returned.
        In case of error, a (None, None) is returned and a warning is issued.
    """
    if cfg is None:
        cfg = get_config()
    if dtshape.shape == ():
        return (), ()
    chunkshape, blockshape = ext.partition_advice(
        dtshape, min_chunksize, max_chunksize, min_blocksize, max_blocksize, cfg
    )
    if chunkshape is None:
        warnings.warn(
            "Error in providing partition advice (please report this)."
            "  Please do not trust on the chunkshape and blockshape in `storage`!",
            UserWarning,
        )
    return chunkshape, blockshape


@dataclass
class DefaultConfig:
    codec: Any
    clevel: Any
    use_dict: Any
    filters: Any
    nthreads: Any
    fp_mantissa_bits: Any
    storage: Any
    eval_method: Any
    seed: Any
    random_gen: Any


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
    # Keep in sync the defaults below with Config.__doc__ docstring.
    _config = None
    codec: ia.Codecs = ia.Codecs.LZ4
    clevel: int = 5
    use_dict: bool = False
    filters: List[ia.Filters] = field(default_factory=default_filters)
    nthreads: int = 0
    fp_mantissa_bits: int = 0
    eval_method: int = ia.Eval.AUTO
    seed: int = None
    random_gen: ia.RandomGen = ia.RandomGen.MERSENNE_TWISTER
    # Storage
    _storage = None
    chunkshape: Sequence = None
    blockshape: Sequence = None
    filename: str = None
    enforce_frame: bool = False
    plainbuffer: bool = False

    def __post_init__(self):
        # Initialize config and storage with its getters and setters
        self.config = self.config

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

    def _random_gen(self):
        return self.random_gen

    @property
    def config(self):
        if self._config is None:
            # Bootstrap the defaults
            return DefaultConfig(
                codec=self.codec,
                clevel=self.clevel,
                use_dict=self.use_dict,
                filters=self.filters,
                nthreads=self.nthreads,
                fp_mantissa_bits=self.fp_mantissa_bits,
                storage=self.storage,
                eval_method=self.eval_method,
                seed=self.seed,
                random_gen=self.random_gen,
            )
        return self._config

    @config.setter
    def config(self, value):
        if not hasattr(value, "codec"):
            raise ValueError(f"You need to use a `Config` instance")
        self.codec = value.codec
        self.clevel = value.clevel
        self.use_dict = value.use_dict
        self.filters = value.filters
        self.nthreads = value.nthreads
        self.fp_mantissa_bits = value.fp_mantissa_bits
        self.eval_method = value.eval_method
        self.seed = value.seed
        self.random_gen = value.random_gen
        self._storage = value.storage
        self._config = value
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
# Global config
global_config = None


def reset_config_defaults():
    """Reset the defaults of the configuration parameters."""
    global global_config
    defaults.config = Defaults()
    global_config = None
    set_config()
    return global_config


@dataclass
class Storage:
    """Dataclass for hosting different storage properties.

    All the parameters below are optional.  In case you don't specify one, a
    sensible default (see below) is used.

    Parameters
    ----------
    chunkshape : list, tuple
        The chunkshape for the output array.  If None (the default), a sensible default
        will be used based on the shape of the array and the size of caches in the current
        processor.
    blockshape : list, tuple
        The blockshape for the output array.  If None (the default), a sensible default
        will be used based on the shape of the array and the size of caches in the current
        processor.
    filename : str
        The name of the file for persistently storing the output array.  If None (the default),
        the output array will be stored in-memory.
    enforce_frame : bool
        If True, the output array will be stored as a frame, even when in-memory.  If False
        (the default), the storage will be sparse.  Currently, persistent storage only supports
        the frame format. When in-memory, the array can be in sparse (the default)
        or contiguous form (frame), depending on this flag.
    plainbuffer : bool
        When True, the output array will be stored on a plain, contiguous buffer, without
        any compression.  This can help faster data sharing among other data containers
        (e.g. NumPy).  When False (the default), the output array will be stored in a Blosc
        container, which can be compressed (the default).
    """

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

    def _get_shape_advice(self, dtshape):
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
class Config(ext.Config):
    """Dataclass for hosting the different ironArray parameters.

    All the parameters below are optional.  In case you don't specify one, a
    sensible default (see below) is used.

    Parameters
    ----------
    codec : Codecs
        The codec to be used inside Blosc.  Default is :py:obj:`Codecs.LZ4 <Codecs>`.
    clevel : int
        The compression level.  It can have values between 0 (no compression) and
        9 (max compression).  Default is 5.
    filters : list
        The list of filters for Blosc.  Default is [:py:obj:`Filters.SHUFFLE <Filters>`].
    fp_mantissa_bits : int
        The number of bits to be kept in the mantissa in output arrays.  If 0 (the default),
        no precision is capped.  FYI, double precision have 52 bit in mantissa, whereas
        single precision has 23 bit.  For example, if you set this to 23 for doubles,
        you will be using a compressed storage very close as if you were using singles.
    use_dict : bool
        Whether Blosc should use a dictionary for enhanced compression (currently only
        supported by :py:obj:`Codecs.ZSTD <Codecs>`).  Default is False.
    nthreads : int
        The number of threads for internal ironArray operations.  This number can be
        silently capped to be the number of *logical* cores in the system.  If 0
        (the default), the number of logical cores in the system is used.
    eval_method : Eval
        Method to evaluate expressions.  The default is :py:obj:`Eval.AUTO <Eval>`, where the
        expression is analyzed and the more convenient method is used.
    seed : int
        The default seed for internal random generators.  If None (the default), a
        seed will automatically be generated internally for you.
    random_gen : RandomGen
        The random generator to be used.  The default is
        :py:obj:`RandomGen.MERSENNE_TWISTER <RandomGen>`.
    storage : Storage
        Storage instance where you can specify different properties of the output
        storage.  See :py:obj:`Storage` docs for details.  For convenience, you can also
        pass all the Storage parameters directly in this constructor too.

    See Also
    --------
    set_config
    config
    """

    codec: ia.Codecs = field(default_factory=defaults._codec)
    clevel: int = field(default_factory=defaults._clevel)
    filters: List[ia.Filters] = field(default_factory=defaults._filters)
    fp_mantissa_bits: int = field(default_factory=defaults._fp_mantissa_bits)
    use_dict: bool = field(default_factory=defaults._use_dict)
    nthreads: int = field(default_factory=defaults._nthreads)
    eval_method: int = field(default_factory=defaults._eval_method)
    seed: int = field(default_factory=defaults._seed)
    random_gen: ia.RandomGen = field(default_factory=defaults._random_gen)
    storage: Storage = None  # delayed initialization

    # These belong to Storage, but we accept them in top level too
    chunkshape: Union[Sequence, None] = field(default_factory=defaults._chunkshape)
    blockshape: Union[Sequence, None] = field(default_factory=defaults._blockshape)
    filename: str = field(default_factory=defaults._filename)
    enforce_frame: bool = field(default_factory=defaults._enforce_frame)
    plainbuffer: bool = field(default_factory=defaults._plainbuffer)

    def __post_init__(self):
        global RANDOM_SEED
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
        self.nthreads = get_ncores(self.nthreads)

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

    def _replace(self, **kwargs):
        cfg_ = replace(self, **kwargs)
        if "storage" in kwargs:
            store = kwargs["storage"]
            if store is not None:
                for field in fields(Storage):
                    setattr(cfg_, field.name, getattr(store, field.name))
        store_args = dict(
            (field.name, kwargs[field.name]) for field in fields(Storage) if field.name in kwargs
        )
        cfg_.storage = replace(cfg_.storage, **store_args)
        return cfg_


def set_config(cfg: Config = None, dtshape=None, **kwargs):
    """Set the global defaults for iarray operations.

    Parameters
    ----------
    cfg : ia.Config
        The configuration that will become the default for iarray operations.
        If None, the defaults are not changed.
    dtshape : ia.DTShape
        This is not part of the global configuration as such, but if passed,
        it will be used so as to compute sensible defaults for storage properties
        like chunkshape and blockshape.  This is mainly meant for internal use.
    kwargs : dict
        A dictionary for setting some or all of the fields in the ia.Config
        dataclass that should override the current configuration.

    Returns
    -------
    ia.Config
        The new global configuration.

    See Also
    --------
    ia.Config
    ia.get_config
    """
    global global_config
    if cfg is None:
        if global_config is None:
            cfg = Config()
        else:
            cfg = global_config

    if kwargs != {}:
        cfg = cfg._replace(**kwargs)
    if dtshape is not None:
        cfg.storage._get_shape_advice(dtshape)

    global_config = cfg
    # Set the defaults for Config() constructor and other nested configs (Storage...)
    defaults.config = cfg

    return global_config


# Initialize the configuration
set_config()


def get_config():
    """Get the global defaults for iarray operations.

    Returns
    -------
    ia.Config
        The existing global configuration.

    See Also
    --------
    ia.set_config
    """
    return global_config


@contextmanager
def config(cfg: Config = None, dtshape=None, **kwargs):
    """Create a context with some specific configuration parameters.

    All parameters are the same than in `ia.set_config()`.
    The only difference is that this does not set global defaults.

    See Also
    --------
    ia.set_config
    ia.Config
    """
    if cfg is None:
        cfg = Config()
    cfg_ = cfg._replace(**kwargs)
    if dtshape is not None:
        cfg_.storage._get_shape_advice(dtshape)
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

    set_config(Config(clevel=1))
    cfg = get_config()
    print("3rd form:", cfg)
    assert cfg.clevel == 1

    with config(clevel=0, enforce_frame=True) as cfg_new:
        print("Context form:", cfg_new)
        assert cfg_new.storage.enforce_frame == True

    cfg = ia.Config(codec=ia.Codecs.BLOSCLZ)
    cfg2 = ia.set_config(cfg=cfg, codec=ia.Codecs.LIZARD)
    print("Standalone config:", cfg)
    print("Global config", cfg2)

    cfg = ia.set_config(cfg_)
    print("Defaults config:", cfg)
