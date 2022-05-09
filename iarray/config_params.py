###########################################################################################
# Copyright ironArray SL 2021.
#
# All rights reserved.
#
# This software is the confidential and proprietary information of ironArray SL
# ("Confidential Information"). You shall not disclose such Confidential Information
# and shall use it only in accordance with the terms of the license agreement.
###########################################################################################

import iarray as ia
from iarray import iarray_ext as ext
from dataclasses import dataclass, field, replace, asdict
from typing import List, Sequence, Any, Union
import warnings
from contextlib import contextmanager
import numpy as np
import copy

# Global variable for random seed
RANDOM_SEED = 0


def get_l2_size():
    """
    Get the L2 size of the system.

    Returns
    -------
    The L2 size discovered in the system.
    """
    l2_size = ext.get_l2_size()
    if l2_size < 0:
        l2_size = 256 * 1024
        warnings.warn(
            "Error getting the l2 size of this system (please report this)."
            f"Returning {l2_size} bytes",
            UserWarning,
        )
    return l2_size


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
    cfg : :class:`Config`
        A configuration.  If None, the global configuration is used.

    Returns
    -------
    tuple
        If success, a (chunks, blocks) containing the advice is returned.
        In case of error, a (None, None) is returned and a warning is issued.
    """
    if cfg is None:
        cfg = get_config_defaults()

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
    zfp_meta: Any
    clevel: Any
    favor: Any
    use_dict: Any
    filters: Any
    nthreads: Any
    fp_mantissa_bits: Any
    eval_method: Any
    seed: Any
    random_gen: Any
    btune: Any
    dtype: Any
    np_dtype: Any
    split_mode: Any


def default_filters():
    return [ia.Filter.SHUFFLE]


@dataclass
class Defaults(object):
    # Config params
    # Keep in sync the defaults below with Config.__doc__ docstring.
    _config = None
    codec: ia.Codec = ia.Codec.LZ4
    zfp_meta: int = 0
    clevel: int = 9
    favor: ia.Favor = ia.Favor.BALANCE
    use_dict: bool = False
    filters: List[ia.Filter] = field(default_factory=default_filters)
    nthreads: int = 0
    fp_mantissa_bits: int = 0
    eval_method: ia.Eval = ia.Eval.AUTO
    seed: int = None
    random_gen: ia.RandomGen = ia.RandomGen.MRG32K3A
    btune: bool = True
    dtype: (
        np.float64,
        np.float32,
        np.int64,
        np.int32,
        np.int16,
        np.int8,
        np.uint64,
        np.uint32,
        np.uint16,
        np.uint8,
        np.bool_,
    ) = np.float64
    np_dtype: bytes or str or np.dtype() = None
    split_mode: (ia.SplitMode) = ia.SplitMode.AUTO_SPLIT
    chunks: Sequence = None
    blocks: Sequence = None
    urlpath: bytes or str = None
    mode: str = "w-"
    contiguous: bool = None

    # Keep track of the special params set with default values for consistency checks with btune
    compat_params: set = field(default_factory=set)
    check_compat: bool = True

    def __post_init__(self):
        # Initialize config and store with its getters and setters
        self.config = self.config

    # Accessors only meant to serve as default_factory
    def _codec(self):
        self.compat_params.add("codec")
        return self.codec

    def _meta(self):
        self.compat_params.add("zfp_meta")
        return self.zfp_meta

    def _clevel(self):
        self.compat_params.add("clevel")
        return self.clevel

    def _favor(self):
        self.compat_params.add("favor")
        return self.favor

    def _use_dict(self):
        return self.use_dict

    def _filters(self):
        self.compat_params.add("filters")
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
        # To check dtype is set when np_dtype != None
        self.compat_params.add("dtype")
        return self.dtype

    def _np_dtype(self):
        # To check dtype is set when np_dtype != None
        self.compat_params.add("np_dtype")
        return self.np_dtype

    def _split_mode(self):
        return self.split_mode

    @property
    def config(self):
        if self._config is None:
            # Bootstrap the defaults
            return DefaultConfig(
                codec=self.codec,
                zfp_meta=self.zfp_meta,
                clevel=self.clevel,
                favor=self.favor,
                use_dict=self.use_dict,
                filters=self.filters,
                nthreads=self.nthreads,
                fp_mantissa_bits=self.fp_mantissa_bits,
                eval_method=self.eval_method,
                seed=self.seed,
                random_gen=self.random_gen,
                btune=self.btune,
                dtype=self.dtype,
                np_dtype=self.np_dtype,
                split_mode=self.split_mode,
            )
        return self._config

    @config.setter
    def config(self, value):
        if not hasattr(value, "codec"):
            raise ValueError(f"You need to use a `Config` instance")
        self.codec = value.codec
        self.zfp_meta = value.zfp_meta
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
        self.np_dtype = value.np_dtype
        self.split_mode = value.split_mode
        self._config = value

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


# Global variable where the defaults for config params are stored
defaults = Defaults()


@dataclass
class Config(ext.Config):
    """Dataclass for hosting the different ironArray parameters.

    All the parameters below are optional.  In case you don't specify one, a
    sensible default (see below) is used.

    Parameters
    ----------
    codec: :class:`Codec`
        The codec to be used inside Blosc.  Default is :py:obj:`Codec.ZSTD <Codec>`.
    zfp_meta: int
        It should be set when using :py:obj:`Codec.ZFP_FIXED_ACCURACY <Codec>`,
        :py:obj:`Codec.ZFP_FIXED_PRECISION <Codec>` or :py:obj:`Codec.ZFP_FIXED_RATE <Codec>`.
        It sets the absolute error, precision or ratio respectively. For more info see
        `C-Blosc2 documentation
        <https://github.com/Blosc/c-blosc2/blob/main/plugins/codecs/zfp/README.md#plugin-usage>`__ .
    clevel : int
        The compression level.  It can have values between 0 (no compression) and
        9 (max compression).  Default is 1.
    btune: bool
        Enable btune machinery. The default is True. When setting :paramref:`favor` `btune`
        has to be enabled, whereas when setting :paramref:`clevel`, :paramref:`codec` or
        :paramref:`filters`, it has to be disabled.
    favor: :class:`Favor`
        What favor when compressing. Possible values are :py:obj:`Favor.SPEED <Favor>`
        for better speed, :py:obj:`Favor.CRATIO <Favor>` for better compression ratios
        and :py:obj:`Favor.BALANCE <Favor>` for a balance among the two.  Default is :py:obj:`Favor.BALANCE <Favor>`.
        For this to work properly, :paramref:`btune` has to be enabled (the default).
    filters : :class:`Filter` list
        The list of filters for Blosc.  Default is [:py:obj:`Filter.BITSHUFFLE <Filter>`].
    fp_mantissa_bits : int
        The number of bits to be kept in the mantissa in output arrays.  If 0 (the default),
        no precision is capped.  FYI, double precision have 52 bit in mantissa, whereas
        single precision has 23 bit.  For example, if you set this to 23 for doubles,
        you will be using a compressed store very close as if you were using singles.
        This automatically enables the ia.Filter.TRUNC_PREC at the front of the
        :paramref:`filters` list.
    use_dict : bool
        Whether Blosc should use a dictionary for enhanced compression (currently only
        supported by :py:obj:`Codec.ZSTD <Codec>`).  Default is False.
    nthreads : int
        The number of threads for internal ironArray operations.  This number can be
        silently capped to be the number of *logical* cores in the system.  If 0
        (the default), the number of logical cores in the system is used.
    eval_method : :class:`Eval`
        Method to evaluate expressions.  The default is :py:obj:`Eval.AUTO <Eval>`, where the
        expression is analyzed and the more convenient method is used.
    seed : int
        The default seed for internal random generators.  If None (the default), a
        seed will automatically be generated internally for you.
    random_gen : :class:`RandomGen`
        The random generator to be used.  The default is
        :py:obj:`RandomGen.MERSENNE_TWISTER <RandomGen>`.
    dtype: (np.float64, np.float32, np.int64, np.int32, np.int16, np.int8, np.uint64, np.uint32, np.uint16,
        np.uint8, np.bool_)
        The data type to use. The default is np.float64.
    np_dtype: bytes, str or np.dtype instance
        The array-protocol typestring of the np.dtype object to use. Default is None. If set, :paramref:`dtype`
        must also be set. The native :ref:`IArray` type used to store data will be :paramref:`dtype`, and
        if needed, a cast or a copy will be made when retrieving so that the output type is :paramref:`np_dtype`.
        Caveat emptor: all the internal operations will use the native :paramref:`dtype`, not :paramref:`np_dtype`,
        so it is easy to shoot in your foot if you expect the reverse thing to happen.
    splitmode: :class:`SplitMode`
        The split mode to be used inside Blosc.  Default is :py:obj:`SplitMode.AUTO_SPLIT <SplitMode>`.
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
        'w-' means create (fail if exists).  Default is 'a' for opening/loading and 'w-' otherwise.
    contiguous : bool
        If True, the output array will be stored contiguously, even when in-memory.  If False,
        the store will be sparse. The default value is False for in-memory and True for persistent
        storage.

    See Also
    --------
    set_config_defaults
    config
    """

    codec: ia.Codec = field(default_factory=defaults._codec)
    zfp_meta: int = field(default_factory=defaults._meta)
    clevel: int = field(default_factory=defaults._clevel)
    favor: ia.Favor = field(default_factory=defaults._favor)
    filters: List[ia.Filter] = field(default_factory=defaults._filters)
    fp_mantissa_bits: int = field(default_factory=defaults._fp_mantissa_bits)
    use_dict: bool = field(default_factory=defaults._use_dict)
    nthreads: int = field(default_factory=defaults._nthreads)
    eval_method: ia.Eval = field(default_factory=defaults._eval_method)
    seed: int = field(default_factory=defaults._seed)
    random_gen: ia.RandomGen = field(default_factory=defaults._random_gen)
    btune: bool = field(default_factory=defaults._btune)
    dtype: (
        np.float64,
        np.float32,
        np.int64,
        np.int32,
        np.int16,
        np.int8,
        np.uint64,
        np.uint32,
        np.uint16,
        np.uint8,
        np.bool_,
    ) = field(default_factory=defaults._dtype)
    np_dtype: bytes or str or np.dtype = field(default_factory=defaults._np_dtype)
    split_mode: ia.SplitMode = field(default_factory=defaults._split_mode)
    chunks: Union[Sequence, None] = field(default_factory=defaults._chunks)
    blocks: Union[Sequence, None] = field(default_factory=defaults._blocks)
    urlpath: bytes or str = field(default_factory=defaults._urlpath)
    mode: str = field(default_factory=defaults._mode)
    contiguous: bool = field(default_factory=defaults._contiguous)

    def __post_init__(self):
        if self.zfp_meta is None:
            self.zfp_meta = 0
        if defaults.check_compat:
            self.check_config_params()
        # Restore variable for next time
        defaults.compat_params = set()
        defaults.check_compat = True

        if self.contiguous is None and self.urlpath is not None:
            self.contiguous = True
        self.urlpath = (
            self.urlpath.encode("utf-8") if isinstance(self.urlpath, str) else self.urlpath
        )
        global RANDOM_SEED
        # Increase the random seed each time so as to prevent re-using them
        if self.seed is None:
            if RANDOM_SEED >= 2 ** 32 - 1:
                # In case we run out of values in uint32_t ints, reset to 0
                RANDOM_SEED = 0
            RANDOM_SEED += 1
            self.seed = RANDOM_SEED

        # Once we have all the settings and hints from the user, we can proceed
        # with some fine tuning.
        # The settings below are based on experiments on a i9-10940X processor.
        if self.nthreads == 0:
            ncores = get_ncores(0)
            # Experiments say that nthreads is optimal when is ~1.5x the number of logical cores
            # self.nthreads = ncores // 2 + ncores // 4
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

        # Activate TRUNC_PREC filter only if mantissa_bits > 0
        if self.fp_mantissa_bits != 0 and ia.Filter.TRUNC_PREC not in self.filters:
            self.filters.insert(0, ia.Filter.TRUNC_PREC)
        # De-activate TRUNC_PREC filter if mantissa_bits == 0
        if self.fp_mantissa_bits == 0 and ia.Filter.TRUNC_PREC in self.filters:
            self.filters.pop(0)
        if self.np_dtype is not None:
            self.np_dtype = np.dtype(self.np_dtype).str

        # Initialize the Cython counterpart
        super().__init__(
            self.codec,
            self.zfp_meta,
            self.clevel,
            self.favor,
            self.use_dict,
            self.filters,
            self.nthreads,
            self.fp_mantissa_bits,
            self.eval_method,
            self.btune,
            self.split_mode,
        )

    def _replace(self, **kwargs):
        # When a replace is done a new object from the class is created with all its params passed as kwargs
        defaults.check_compat = False
        cfg_ = replace(self, **kwargs)
        return cfg_

    def __deepcopy__(self, memodict={}):
        kwargs = asdict(self)
        defaults.check_compat = False
        cfg = Config(**kwargs)
        return cfg

    def check_config_params(self, **kwargs):
        # Check incompatibilities
        # dtype set when np_dtype also set
        if kwargs != {} and "np_dtype" in kwargs:
            np_dtype_allowed = all(x in kwargs for x in ["np_dtype", "dtype"])
        else:
            np_dtype_allowed = all(x not in defaults.compat_params for x in ["np_dtype", "dtype"])
        np_dtype = kwargs.get("np_dtype", self.np_dtype)
        if np_dtype is not None and not np_dtype_allowed:
            defaults.compat_params = set()
            defaults.check_compat = True
            raise ValueError("`dtype` must be explicitly set when setting `np_dtype`")

        # btune=True with others
        btune = kwargs.get("btune", self.btune)

        btune_incompatible = {"clevel", "codec", "filters"}
        if kwargs != {}:
            btune_allowed = all(x not in kwargs for x in btune_incompatible)
        else:
            btune_allowed = all(x in defaults.compat_params for x in btune_incompatible)
        if btune and not btune_allowed:
            # Restore variable for next time
            defaults.compat_params = set()
            defaults.check_compat = True
            raise ValueError(
                f"To set any flag in",
                btune_incompatible,
                "you need to disable `btune` explicitly.",
            )

        # favor=something and btune=False
        if kwargs != {}:
            if "favor" in kwargs and not btune:
                # Restore variable for next time
                defaults.compat_params = set()
                defaults.check_compat = True
                raise ValueError(f"A `favor` argument needs `btune` enabled.")
        else:
            if "favor" not in defaults.compat_params and not btune:
                # Restore variable for next time
                defaults.compat_params = set()
                defaults.check_compat = True
                raise ValueError(f"A `favor` argument needs `btune` enabled.")

        # codec = ZFP and zfp_meta = None
        zfp_meta = kwargs.get("zfp_meta", self.zfp_meta)
        codec = kwargs.get("codec", self.codec)
        zfp_codecs = [ia.Codec.ZFP_FIXED_PRECISION, ia.Codec.ZFP_FIXED_RATE, ia.Codec.ZFP_FIXED_ACCURACY]
        if codec in zfp_codecs:
            if zfp_meta is None or zfp_meta == 0:
                # Restore variable for next time
                defaults.compat_params = set()
                defaults.check_compat = True
                raise ValueError(f"`zfp_meta` needs to be set when using a ZFP codec.")
            filters = kwargs.get("filters", self.filters)
            if filters != [ia.Filter.NOFILTER]:
                # Restore variable for next time
                defaults.compat_params = set()
                defaults.check_compat = True
                raise ValueError(f"`filters` must be `[ia.Filter.NOFILTER]` when using a ZFP codec.")
        elif zfp_meta is not None and zfp_meta != 0:
            # Restore variable for next time
            defaults.compat_params = set()
            defaults.check_compat = True
            raise ValueError(f"`zfp_meta` can only be set when using a ZFP codec.")

    def _get_shape_advice(self, shape):
        chunks, blocks = self.chunks, self.blocks
        if chunks is not None and blocks is not None:
            return
        if chunks is None and blocks is None:
            chunks_, blocks_ = partition_advice(shape, cfg=self)
            self.chunks = chunks_
            self.blocks = blocks_
            return
        else:
            raise ValueError("You can either specify both chunks and blocks or none of them.")


# Global config
global_config = Config()


def get_config_defaults():
    """Get the global defaults for iarray operations.

    Returns
    -------
    :class:`Config`
        The existing global configuration.

    See Also
    --------
    set_config_defaults
    """
    return global_config


def set_config_defaults(cfg: Config = None, shape=None, **kwargs):
    """Set the global defaults for iarray operations.

    Parameters
    ----------
    cfg : :class:`Config`
        The configuration that will become the default for iarray operations.
        If None, the defaults are not changed.
    shape : Sequence
        This is not part of the global configuration as such, but if passed,
        it will be used so as to compute sensible defaults for store properties
        like chunk shape and block shape.  This is mainly meant for internal use.
    kwargs : dict
        A dictionary for setting some or all of the fields in the :class:`Config`
        dataclass that should override the current configuration.

    Returns
    -------
    :class:`Config`
        The new global configuration.

    See Also
    --------
    Config
    get_config_defaults
    """
    global global_config
    global defaults

    cfg_old = get_config_defaults()

    if cfg is None:
        cfg = copy.deepcopy(cfg_old)
    else:
        cfg = copy.deepcopy(cfg)

    if kwargs != {}:
        cfg.check_config_params(**kwargs)
        # The default when creating frames on-disk is to use contiguous storage (mainly because of performance  reasons)
        if (
            kwargs.get("contiguous", None) is None
            and cfg.contiguous is None
            and kwargs.get("urlpath", None) is not None
        ):
            cfg = cfg._replace(**dict(kwargs, contiguous=True))
        else:
            cfg = cfg._replace(**kwargs)
    if shape is not None and cfg.chunks is None and cfg.blocks is None:
        cfg._get_shape_advice(shape)

    global_config = cfg
    defaults.config = cfg

    return get_config_defaults()


# Initialize the configuration


@contextmanager
def config(cfg: Config = None, shape=None, **kwargs):
    """Create a context with some specific configuration parameters.

    All parameters are the same than in :class:`Config()`.
    The only difference is that this does not set global defaults.

    See Also
    --------
    set_config_defaults
    Config
    """
    global global_config
    global defaults

    cfg_aux = ia.get_config_defaults()
    cfg = set_config_defaults(cfg, shape, **kwargs)

    try:
        yield cfg
    finally:
        defaults.config = cfg_aux
        global_config = cfg_aux


def reset_config_defaults():
    """Reset the defaults of the configuration parameters."""
    global global_config
    global defaults

    defaults.config = Defaults()
    global_config = Config()
    return global_config


if __name__ == "__main__":
    cfg_ = get_config_defaults()
    print("Defaults:", cfg_)
    assert cfg_.contiguous is False

    set_config_defaults(contiguous=True)
    cfg = get_config_defaults()
    print("1st form:", cfg)
    assert cfg.contiguous is True

    set_config_defaults(contiguous=False)
    cfg = get_config_defaults()
    print("2nd form:", cfg)
    assert cfg.contiguous is False

    set_config_defaults(Config(clevel=5))
    cfg = get_config_defaults()
    print("3rd form:", cfg)
    assert cfg.clevel == 5

    with config(clevel=0, contiguous=True) as cfg_new:
        print("Context form:", cfg_new)
        assert cfg_new.contiguous is True
        assert get_config_defaults().clevel == 0

    cfg = ia.Config(codec=ia.Codec.BLOSCLZ)
    cfg2 = ia.set_config_defaults(cfg=cfg, codec=ia.Codec.LIZARD)
    print("Standalone config:", cfg)
    print("Global config", cfg2)

    cfg = ia.set_config_defaults(cfg_)
    print("Defaults config:", cfg)
