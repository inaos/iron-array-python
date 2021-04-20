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
import numpy as np

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
    shape, min_chunksize=0, max_chunksize=0, min_blocksize=0, max_blocksize=0, cfg=None
):
    """Provide advice for the chunk and block shapes for a certain `dtshape`.

    Parameters
    ----------
    shape : Sequence
        The shape of the array.
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

    dtshape = ia.DTShape(shape, cfg.dtype)
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
    favor: Any
    use_dict: Any
    filters: Any
    nthreads: Any
    fp_mantissa_bits: Any
    store: Any
    eval_method: Any
    seed: Any
    random_gen: Any
    btune: Any
    dtype: Any


@dataclass
class DefaultStore:
    chunkshape: Any
    blockshape: Any
    urlpath: Any
    enforce_frame: Any
    plainbuffer: Any


def default_filters():
    return [ia.Filters.BITSHUFFLE]


@dataclass
class Defaults(object):
    # Config params
    # Keep in sync the defaults below with Config.__doc__ docstring.
    _config = None
    codec: ia.Codecs = ia.Codecs.ZSTD
    clevel: int = 1
    favor: ia.Favors = ia.Favors.BALANCE
    use_dict: bool = False
    filters: List[ia.Filters] = field(default_factory=default_filters)
    nthreads: int = 0
    fp_mantissa_bits: int = 0
    eval_method: int = ia.Eval.AUTO
    seed: int = None
    random_gen: ia.RandomGen = ia.RandomGen.MERSENNE_TWISTER
    btune: bool = True
    dtype: (np.float32, np.float64) = np.float64

    # Store
    _store = None
    chunkshape: Sequence = None
    blockshape: Sequence = None
    urlpath: str = None
    enforce_frame: bool = False
    plainbuffer: bool = False

    def __post_init__(self):
        # Initialize config and store with its getters and setters
        self.config = self.config

    # Accessors only meant to serve as default_factory
    def _codec(self):
        return self.codec

    def _clevel(self):
        return self.clevel

    def _favor(self):
        return self.favor

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

    def _btune(self):
        return self.btune

    def _dtype(self):
        return self.dtype

    @property
    def config(self):
        if self._config is None:
            # Bootstrap the defaults
            return DefaultConfig(
                codec=self.codec,
                clevel=self.clevel,
                favor=self.favor,
                use_dict=self.use_dict,
                filters=self.filters,
                nthreads=self.nthreads,
                fp_mantissa_bits=self.fp_mantissa_bits,
                store=self.store,
                eval_method=self.eval_method,
                seed=self.seed,
                random_gen=self.random_gen,
                btune=self.btune,
                dtype=self.dtype,
            )
        return self._config

    @config.setter
    def config(self, value):
        if not hasattr(value, "codec"):
            raise ValueError(f"You need to use a `Config` instance")
        self.codec = value.codec
        self.clevel = value.clevel
        self.favor = value.favor
        self.use_dict = value.use_dict
        self.filters = value.filters
        self.nthreads = value.nthreads
        self.fp_mantissa_bits = value.fp_mantissa_bits
        self.eval_method = value.eval_method
        self.seed = value.seed
        self.random_gen = value.random_gen
        self.btune = value.btune
        self.dtype = value.dtype
        self._store = value.store
        self._config = value
        if self._store is not None:
            self.set_store(self._store)

    def _chunkshape(self):
        return self.chunkshape

    def _blockshape(self):
        return self.blockshape

    def _urlpath(self):
        return self.urlpath

    def _enforce_frame(self):
        return self.enforce_frame

    def _plainbuffer(self):
        return self.plainbuffer

    @property
    def store(self):
        if self._store is None:
            # Bootstrap the defaults
            return DefaultStore(
                chunkshape=self.chunkshape,
                blockshape=self.blockshape,
                urlpath=self.urlpath,
                enforce_frame=self.enforce_frame,
                plainbuffer=self.plainbuffer,
            )
        return self._store

    def set_store(self, value):
        if not hasattr(value, "chunkshape"):
            raise ValueError(f"You need to use a `Store` instance")
        self.chunkshape = value.chunkshape
        self.blockshape = value.blockshape
        self.urlpath = value.urlpath
        self.enforce_frame = value.enforce_frame
        self.plainbuffer = value.plainbuffer
        self._store = value


# Global variable where the defaults for config params are stored
defaults = Defaults()
# Global config
global_config = []


def reset_config_defaults():
    """Reset the defaults of the configuration parameters."""
    global global_config
    defaults.config = Defaults()
    global_config = []
    set_config()
    return global_config


@dataclass
class Store:
    """Dataclass for hosting different store properties.

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
    urlpath : str
        The name of the file for persistently storing the output array.  If None (the default),
        the output array will be stored in-memory.
    enforce_frame : bool
        If True, the output array will be stored as a frame, even when in-memory.  If False
        (the default), the store will be sparse.  Currently, persistent store only supports
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
    urlpath: bytes or str = field(default_factory=defaults._urlpath)
    enforce_frame: bool = field(default_factory=defaults._enforce_frame)
    plainbuffer: bool = field(default_factory=defaults._plainbuffer)

    def __post_init__(self):
        self.urlpath = (
            self.urlpath.encode("utf-8") if isinstance(self.urlpath, str) else self.urlpath
        )
        self.enforce_frame = True if self.urlpath else self.enforce_frame
        if self.plainbuffer:
            if self.chunkshape is not None or self.blockshape is not None:
                raise ValueError(
                    "plainbuffer array does not support neither a chunkshape nor blockshape"
                )

    def _get_shape_advice(self, shape, cfg=None):
        if self.plainbuffer:
            return
        chunkshape, blockshape = self.chunkshape, self.blockshape
        if chunkshape is not None and blockshape is not None:
            return
        if chunkshape is None and blockshape is None:
            chunkshape_, blockshape_ = partition_advice(shape, cfg=cfg)
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
        The codec to be used inside Blosc.  Default is :py:obj:`Codecs.ZSTD <Codecs>`.
    clevel : int
        The compression level.  It can have values between 0 (no compression) and
        9 (max compression).  Default is 1.
    favor : Favors
        What favor when compressing. Possible values are :py:obj:`Favors.SPEED <Favors>`
        for better speed, :py:obj:`Favors.CRATIO <Favors>` for bettwer compresion ratios
        and :py:obj:`Favors.BALANCE <Favors>`.  Default is :py:obj:`Favors.BALANCE <Favors>`.
    filters : list
        The list of filters for Blosc.  Default is [:py:obj:`Filters.BITSHUFFLE <Filters>`].
    fp_mantissa_bits : int
        The number of bits to be kept in the mantissa in output arrays.  If 0 (the default),
        no precision is capped.  FYI, double precision have 52 bit in mantissa, whereas
        single precision has 23 bit.  For example, if you set this to 23 for doubles,
        you will be using a compressed store very close as if you were using singles.
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
    btune: bool
        Enable btune machinery. The default is True.
    dtype: (np.float32, np.float64)
        The data type to use. The default is np.float64.
    store : Store
        Store instance where you can specify different properties of the output
        store.  See :py:obj:`Store` docs for details.  For convenience, you can also
        pass all the Store parameters directly in this constructor too.

    See Also
    --------
    set_config
    config
    """

    codec: ia.Codecs = field(default_factory=defaults._codec)
    clevel: int = field(default_factory=defaults._clevel)
    favor: int = field(default_factory=defaults._favor)
    filters: List[ia.Filters] = field(default_factory=defaults._filters)
    fp_mantissa_bits: int = field(default_factory=defaults._fp_mantissa_bits)
    use_dict: bool = field(default_factory=defaults._use_dict)
    nthreads: int = field(default_factory=defaults._nthreads)
    eval_method: int = field(default_factory=defaults._eval_method)
    seed: int = field(default_factory=defaults._seed)
    random_gen: ia.RandomGen = field(default_factory=defaults._random_gen)
    btune: bool = field(default_factory=defaults._btune)
    dtype: (np.float32, np.float64) = field(default_factory=defaults._dtype)
    store: Store = None  # delayed initialization

    # These belong to Store, but we accept them in top level too
    chunkshape: Union[Sequence, None] = field(default_factory=defaults._chunkshape)
    blockshape: Union[Sequence, None] = field(default_factory=defaults._blockshape)
    urlpath: bytes or str = field(default_factory=defaults._urlpath)
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
        if self.store is None:
            self.store = Store(
                chunkshape=self.chunkshape,
                blockshape=self.blockshape,
                urlpath=self.urlpath,
                enforce_frame=self.enforce_frame,
                plainbuffer=self.plainbuffer,
            )
        # Once we have all the settings and hints from the user, we can proceed
        # with some fine tuning.
        # The settings below are based on experiments on a i9-10940X processor.
        if self.nthreads == 0:
            ncores = get_ncores(0)
            # Experiments say that nthreads is optimal when is ~1.5x the number of logical cores
            self.nthreads = ncores // 2 + ncores // 4
        if self.favor == ia.Favors.SPEED:
            self.codec = ia.Codecs.LZ4 if self.codec == Defaults.codec else self.codec
            self.clevel = 9 if self.clevel == Defaults.clevel else self.clevel
            self.filters = (
                [ia.Filters.SHUFFLE] if self.filters == default_filters() else self.filters
            )
        elif self.favor == ia.Favors.CRATIO:
            self.codec = ia.Codecs.ZSTD if self.codec == Defaults.codec else self.codec
            self.clevel = 5 if self.clevel == Defaults.clevel else self.clevel
            self.filters = (
                [ia.Filters.BITSHUFFLE] if self.filters == default_filters() else self.filters
            )

        # Initialize the Cython counterpart
        super().__init__(
            self.codec,
            self.clevel,
            self.favor,
            self.use_dict,
            self.filters,
            self.nthreads,
            self.fp_mantissa_bits,
            self.eval_method,
            self.btune,
        )

    def _replace(self, **kwargs):
        cfg_ = replace(self, **kwargs)
        if "store" in kwargs:
            store = kwargs["store"]
            if store is not None:
                for field in fields(Store):
                    setattr(cfg_, field.name, getattr(store, field.name))
        store_args = dict(
            (field.name, kwargs[field.name]) for field in fields(Store) if field.name in kwargs
        )
        cfg_.store = replace(cfg_.store, **store_args)
        return cfg_


def set_config(cfg: Config = None, shape=None, **kwargs):
    """Set the global defaults for iarray operations.

    Parameters
    ----------
    cfg : ia.Config
        The configuration that will become the default for iarray operations.
        If None, the defaults are not changed.
    shape : Sequence
        This is not part of the global configuration as such, but if passed,
        it will be used so as to compute sensible defaults for store properties
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
        if not global_config:
            cfg = Config()
        else:
            cfg = global_config.pop()

    if kwargs != {}:
        cfg = cfg._replace(**kwargs)
    if shape is not None:
        cfg.store._get_shape_advice(shape, cfg=cfg)

    global_config.append(cfg)
    # Set the defaults for Config() constructor and other nested configs (Store...)
    defaults.config = cfg

    return global_config[-1]


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
    global global_config

    return global_config[-1]


@contextmanager
def config(cfg: Config = None, shape=None, **kwargs):
    """Create a context with some specific configuration parameters.

    All parameters are the same than in `ia.set_config()`.
    The only difference is that this does not set global defaults.

    See Also
    --------
    ia.set_config
    ia.Config
    """
    global global_config

    if cfg is None:
        cfg = Config()
    cfg_ = cfg._replace(**kwargs)
    if shape is not None:
        cfg_.store._get_shape_advice(shape)
    global_config.append(cfg_)

    try:
        yield cfg_
    finally:
        global_config.pop()


if __name__ == "__main__":
    cfg_ = get_config()
    print("Defaults:", cfg_)
    assert cfg_.store.enforce_frame is False

    set_config(store=Store(enforce_frame=True))
    cfg = get_config()
    print("1st form:", cfg)
    assert cfg.store.enforce_frame is True

    set_config(enforce_frame=False)
    cfg = get_config()
    print("2nd form:", cfg)
    assert cfg.store.enforce_frame is False

    set_config(Config(clevel=5))
    cfg = get_config()
    print("3rd form:", cfg)
    assert cfg.clevel == 5

    with config(clevel=0, enforce_frame=True) as cfg_new:
        print("Context form:", cfg_new)
        assert cfg_new.store.enforce_frame is True
        assert get_config().clevel == 0

    cfg = ia.Config(codec=ia.Codecs.BLOSCLZ)
    cfg2 = ia.set_config(cfg=cfg, codec=ia.Codecs.LIZARD)
    print("Standalone config:", cfg)
    print("Global config", cfg2)

    cfg = ia.set_config(cfg_)
    print("Defaults config:", cfg)
