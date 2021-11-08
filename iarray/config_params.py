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
from dataclasses import dataclass, field, fields, replace, asdict
from typing import List, Sequence, Any, Union
import warnings
from contextlib import contextmanager
import numpy as np
import copy

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
        If success, a (chunks, blocks) containing the advice is returned.
        In case of error, a (None, None) is returned and a warning is issued.
    """
    if cfg is None:
        cfg = get_config()

    dtshape = ia.DTShape(shape, cfg.dtype)
    if dtshape.shape == ():
        return (), ()
    chunks, blocks = ext.partition_advice(
        dtshape, min_chunksize, max_chunksize, min_blocksize, max_blocksize, cfg
    )
    if chunks is None:
        warnings.warn(
            "Error in providing partition advice (please report this)."
            "  Please do not trust on the chunks and blocks in `storage`!",
            UserWarning,
        )
    return chunks, blocks


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
    chunks: Any
    blocks: Any
    urlpath: Any
    mode: Any
    contiguous: Any
    plainbuffer: Any


def default_filters():
    return [ia.Filter.SHUFFLE]


@dataclass
class Defaults(object):
    # Config params
    # Keep in sync the defaults below with Config.__doc__ docstring.
    _config = None
    codec: ia.Codec = ia.Codec.LZ4
    clevel: int = 9
    favor: ia.Favor = ia.Favor.BALANCE
    use_dict: bool = False
    filters: List[ia.Filter] = field(default_factory=default_filters)
    nthreads: int = 0
    fp_mantissa_bits: int = 0
    eval_method: int = ia.Eval.AUTO
    seed: int = None
    random_gen: ia.RandomGen = ia.RandomGen.MERSENNE_TWISTER
    btune: bool = True
    dtype: (np.float32, np.float64) = np.float64

    # Store
    _store = None
    chunks: Sequence = None
    blocks: Sequence = None
    urlpath: str = None
    mode: str = "r"

    contiguous: bool = None
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

    def _chunks(self):
        return self.chunks

    def _blocks(self):
        return self.blocks

    def _urlpath(self):
        return self.urlpath

    def _mode(self):
        return self.mode

    def _contiguous(self):
        return self.contiguous

    def _plainbuffer(self):
        return self.plainbuffer

    @property
    def store(self):
        if self._store is None:
            # Bootstrap the defaults
            return DefaultStore(
                chunks=self.chunks,
                blocks=self.blocks,
                urlpath=self.urlpath,
                mode=self.mode,
                contiguous=self.contiguous,
                plainbuffer=self.plainbuffer,
            )
        return self._store

    def set_store(self, value):
        if not hasattr(value, "chunks"):
            raise ValueError(f"You need to use a `Store` instance")
        self.chunks = value.chunks
        self.blocks = value.blocks
        self.urlpath = value.urlpath
        self.mode = value.mode
        self.contiguous = value.contiguous
        self.plainbuffer = value.plainbuffer
        self._store = value


# Global variable where the defaults for config params are stored
defaults = Defaults()


@dataclass
class Store:
    """Dataclass for hosting different store properties.

    All the parameters below are optional.  In case you don't specify one, a
    sensible default (see below) is used.

    Parameters
    ----------
    chunks : list, tuple
        The chunk shape for the output array.  If None (the default), a sensible default
        will be used based on the shape of the array and the size of caches in the current
        processor.
    blocks : list, tuple
        The block shape for the output array.  If None (the default), a sensible default
        will be used based on the shape of the array and the size of caches in the current
        processor.
    urlpath : str
        The name of the file for persistently storing the output array.  If None (the default),
        the output array will be stored in-memory.
    mode : str
        Persistence mode: 'r' means read only (must exist); 'r+' means read/write (must exist);
        'a' means read/write (create if doesn’t exist); 'w' means create (overwrite if exists);
        'w-' means create (fail if exists).  Default is 'r'.
    contiguous : bool
        If True, the output array will be stored contiguously, even when in-memory.  If False,
        the store will be sparse. The default value is False for in-memory and True for persistent
        storage.
    plainbuffer : bool
        When True, the output array will be stored on a plain, contiguous buffer, without
        any compression.  This can help faster data sharing among other data containers
        (e.g. NumPy).  When False (the default), the output array will be stored in a Blosc
        container, which can be compressed (the default).
    """

    global defaults
    chunks: Union[Sequence, None] = field(default_factory=defaults._chunks)
    blocks: Union[Sequence, None] = field(default_factory=defaults._blocks)
    urlpath: bytes or str = field(default_factory=defaults._urlpath)
    mode: bytes or str = field(default_factory=defaults._mode)
    contiguous: bool = field(default_factory=defaults._contiguous)
    plainbuffer: bool = field(default_factory=defaults._plainbuffer)

    def __post_init__(self):
        self.urlpath = (
            self.urlpath.encode("utf-8") if isinstance(self.urlpath, str) else self.urlpath
        )
        if self.contiguous is None and self.urlpath is not None:
            self.contiguous = True
        else:
            self.contiguous = self.contiguous
        self.mode = (
            self.mode.encode("utf-8") if isinstance(self.mode, str) else self.mode
        )
        if self.plainbuffer:
            if self.chunks is not None or self.blocks is not None:
                raise ValueError("plainbuffer array does not support neither a chunks nor blocks")

    def _get_shape_advice(self, shape, cfg=None):
        if self.plainbuffer:
            return
        chunks, blocks = self.chunks, self.blocks
        if chunks is not None and blocks is not None:
            return
        if chunks is None and blocks is None:
            chunks_, blocks_ = partition_advice(shape, cfg=cfg)
            self.chunks = chunks_
            self.blocks = blocks_
            return
        else:
            raise ValueError("You can either specify both chunks and blocks or none of them.")


@dataclass
class Config(ext.Config):
    """Dataclass for hosting the different ironArray parameters.

    All the parameters below are optional.  In case you don't specify one, a
    sensible default (see below) is used.

    Parameters
    ----------
    codec : Codec
        The codec to be used inside Blosc.  Default is :py:obj:`Codec.ZSTD <Codec>`.
    clevel : int
        The compression level.  It can have values between 0 (no compression) and
        9 (max compression).  Default is 1.
    favor : Favor
        What favor when compressing. Possible values are :py:obj:`Favor.SPEED <Favor>`
        for better speed, :py:obj:`Favor.CRATIO <Favor>` for better compression ratios
        and :py:obj:`Favor.BALANCE <Favor>` for a balance among the two.  Default is :py:obj:`Favor.BALANCE <Favor>`.
        For this to work properly, `btune` has to be activated (the default).
    filters : list
        The list of filters for Blosc.  Default is [:py:obj:`Filter.BITSHUFFLE <Filter>`].
    fp_mantissa_bits : int
        The number of bits to be kept in the mantissa in output arrays.  If 0 (the default),
        no precision is capped.  FYI, double precision have 52 bit in mantissa, whereas
        single precision has 23 bit.  For example, if you set this to 23 for doubles,
        you will be using a compressed store very close as if you were using singles.
    use_dict : bool
        Whether Blosc should use a dictionary for enhanced compression (currently only
        supported by :py:obj:`Codec.ZSTD <Codec>`).  Default is False.
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

    codec: ia.Codec = field(default_factory=defaults._codec)
    clevel: int = field(default_factory=defaults._clevel)
    favor: int = field(default_factory=defaults._favor)
    filters: List[ia.Filter] = field(default_factory=defaults._filters)
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
    chunks: Union[Sequence, None] = field(default_factory=defaults._chunks)
    blocks: Union[Sequence, None] = field(default_factory=defaults._blocks)
    urlpath: bytes or str = field(default_factory=defaults._urlpath)
    mode: bytes or str = field(default_factory=defaults._mode)
    contiguous: bool = field(default_factory=defaults._contiguous)
    plainbuffer: bool = field(default_factory=defaults._plainbuffer)

    def __post_init__(self):
        if self.urlpath is not None and self.contiguous is None:
            self.contiguous = True
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
                chunks=self.chunks,
                blocks=self.blocks,
                urlpath=self.urlpath,
                mode=self.mode,
                contiguous=self.contiguous,
                plainbuffer=self.plainbuffer,
            )
        # Once we have all the settings and hints from the user, we can proceed
        # with some fine tuning.
        # The settings below are based on experiments on a i9-10940X processor.
        if self.nthreads == 0:
            ncores = get_ncores(0)
            # Experiments say that nthreads is optimal when is ~1.5x the number of logical cores
            #self.nthreads = ncores // 2 + ncores // 4
            # More experiments with AMD 5950X seems to say that using all logical cores is better
            self.nthreads = ncores
        if self.favor == ia.Favor.SPEED:
            self.codec = ia.Codec.LZ4 if self.codec == Defaults.codec else self.codec
            self.clevel = 9 if self.clevel == Defaults.clevel else self.clevel
            self.filters = (
                [ia.Filter.SHUFFLE] if self.filters == default_filters() else self.filters
            )
        elif self.favor == ia.Favor.CRATIO:
            self.codec = ia.Codec.ZSTD if self.codec == Defaults.codec else self.codec
            self.clevel = 5 if self.clevel == Defaults.clevel else self.clevel
            self.filters = (
                [ia.Filter.BITSHUFFLE] if self.filters == default_filters() else self.filters
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
        else:  # avoid overwriting the store
            store_args = dict(
                (field.name, kwargs[field.name]) for field in fields(Store) if field.name in kwargs
            )
            cfg_.store = replace(cfg_.store, **store_args)
        return cfg_

    def __deepcopy__(self, memodict={}):
        kwargs = asdict(self)
        # asdict is recursive, but we need the store kwarg as a Store object
        kwargs["store"] = Store(**kwargs["store"])
        cfg = Config(**kwargs)
        return cfg


# Global config
global_config = Config()
global_diff = []


def get_config(cfg=None):
    """Get the global defaults for iarray operations.

    Parameters
    ----------
    cfg
        The base configuration to which the changes will apply.

    Returns
    -------
    ia.Config
        The existing global configuration.

    See Also
    --------
    ia.set_config
    """
    global global_config
    global global_diff

    if not cfg:
        cfg = global_config
    else:
        cfg = copy.deepcopy(cfg)

    for diff in global_diff:
        cfg = cfg._replace(**diff)

    return cfg


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
        like chunk shape and block shape.  This is mainly meant for internal use.
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
    global global_diff
    global defaults

    cfg_old = get_config()
    d_old = asdict(cfg_old)

    if cfg is None:
        cfg = copy.deepcopy(cfg_old)
    else:
        cfg = copy.deepcopy(cfg)

    if kwargs != {}:
        # The default when creating frames on-disk is to use contiguous storage (mainly because of performance  reasons)
        if kwargs.get('contiguous', None) is None and cfg.contiguous is None and kwargs.get('urlpath', None) is not None:
            cfg = cfg._replace(**dict(kwargs, contiguous=True))
        else:
            cfg = cfg._replace(**kwargs)
    if shape is not None:
        cfg.store._get_shape_advice(shape, cfg=cfg)
        cfg._replace(**{"store": cfg.store})

    d = asdict(cfg)

    diff = {k: d[k] for k in d.keys() if d_old[k] != d[k]}
    if "store" in diff:
        diff["store"] = Store(**diff["store"])

    global_diff.append(diff)
    defaults.config = cfg

    return get_config()


# Initialize the configuration


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
    global global_diff
    global defaults

    cfg_aux = ia.get_config()
    cfg = set_config(cfg, shape, **kwargs)

    try:
        yield cfg
    finally:
        global_diff.pop()
        defaults.config = cfg_aux


def reset_config_defaults():
    """Reset the defaults of the configuration parameters."""
    global global_config
    global global_diff
    global defaults

    defaults.config = Defaults()
    global_config = Config()
    global_diff = []
    return global_config


if __name__ == "__main__":
    cfg_ = get_config()
    print("Defaults:", cfg_)
    assert cfg_.store.contiguous is False

    set_config(store=Store(contiguous=True))
    cfg = get_config()
    print("1st form:", cfg)
    assert cfg.store.contiguous is True

    set_config(contiguous=False)
    cfg = get_config()
    print("2nd form:", cfg)
    assert cfg.store.contiguous is False

    set_config(Config(clevel=5))
    cfg = get_config()
    print("3rd form:", cfg)
    assert cfg.clevel == 5

    with config(clevel=0, contiguous=True) as cfg_new:
        print("Context form:", cfg_new)
        assert cfg_new.store.contiguous is True
        assert get_config().clevel == 0

    cfg = ia.Config(codec=ia.Codec.BLOSCLZ)
    cfg2 = ia.set_config(cfg=cfg, codec=ia.Codec.LIZARD)
    print("Standalone config:", cfg)
    print("Global config", cfg2)

    cfg = ia.set_config(cfg_)
    print("Defaults config:", cfg)
