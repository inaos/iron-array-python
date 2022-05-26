###########################################################################################
# Copyright ironArray SL 2021.
#
# All rights reserved.
#
# This software is the confidential and proprietary information of ironArray SL
# ("Confidential Information"). You shall not disclose such Confidential Information
# and shall use it only in accordance with the terms of the license agreement.
###########################################################################################

from libcpp cimport bool

cdef extern from "<stdint.h>":
    ctypedef   signed char  int8_t
    ctypedef   signed short int16_t
    ctypedef   signed int   int32_t
    ctypedef   signed long  int64_t
    ctypedef unsigned char  uint8_t
    ctypedef unsigned short uint16_t
    ctypedef unsigned int   uint32_t
    ctypedef unsigned long long uint64_t


cdef extern from "libiarray/iarray.h":
    ctypedef uint64_t ina_rc_t

    # Stuff from "libinac/ina.h"

    cdef enum:
        INA_ES_USER_DEFINED = 1024UL

    cdef enum:
        INA_RC_BIT_C = 16U

    cdef enum:
        INA_ERR_COMPILED            = 18ULL << INA_RC_BIT_C
        INA_ERR_FAILED              = 43ULL << INA_RC_BIT_C
        INA_ERR_OUT_OF_RANGE        = 84ULL << INA_RC_BIT_C
    # End of stuff from "libinac/ina.h"

    cdef enum:
        IARRAY_COMP_SHUFFLE = 0x1,
        IARRAY_COMP_BITSHUFFLE = 0x2,
        IARRAY_COMP_DELTA = 0x4,
        IARRAY_COMP_TRUNC_PREC = 0x8,

    cdef enum:
        IARRAY_ES_CONTAINER = INA_ES_USER_DEFINED + 1
        IARRAY_ES_DTSHAPE
        IARRAY_ES_SHAPE
        IARRAY_ES_CHUNKSHAPE
        IARRAY_ES_NDIM
        IARRAY_ES_DTYPE
        IARRAY_ES_STORAGE
        IARRAY_ES_PERSISTENCY
        IARRAY_ES_BUFFER
        IARRAY_ES_CATERVA
        IARRAY_ES_BLOSC
        IARRAY_ES_ASSERTION
        IARRAY_ES_BLOCKSHAPE
        IARRAY_ES_RNG_METHOD
        IARRAY_ES_RAND_METHOD
        IARRAY_ES_RAND_PARAM
        IARRAY_ES_ITER
        IARRAY_ES_EVAL_METHOD
        IARRAY_ES_EVAL_ENGINE

    cdef enum:
        IARRAY_ERR_EVAL_ENGINE_FAILED = INA_ERR_FAILED | IARRAY_ES_EVAL_ENGINE
        IARRAY_ERR_EVAL_ENGINE_NOT_COMPILED = INA_ERR_COMPILED | IARRAY_ES_EVAL_ENGINE
        IARRAY_ERR_EVAL_ENGINE_OUT_OF_RANGE = INA_ERR_OUT_OF_RANGE | IARRAY_ES_EVAL_ENGINE

    cdef enum:
        IARRAY_DIMENSION_MAX = 8

    cdef enum:
        IARRAY_EXPR_OPERANDS_MAX = 128

    cdef enum:
        IARRAY_EXPR_USER_PARAMS_MAX = 128

    ctypedef enum iarray_container_flags_t:
        IARRAY_CONTAINER_PERSIST = 0x1

    ctypedef enum iarray_data_type_t:
        IARRAY_DATA_TYPE_DOUBLE
        IARRAY_DATA_TYPE_FLOAT
        IARRAY_DATA_TYPE_INT64
        IARRAY_DATA_TYPE_INT32
        IARRAY_DATA_TYPE_INT16
        IARRAY_DATA_TYPE_INT8
        IARRAY_DATA_TYPE_UINT64
        IARRAY_DATA_TYPE_UINT32
        IARRAY_DATA_TYPE_UINT16
        IARRAY_DATA_TYPE_UINT8
        IARRAY_DATA_TYPE_BOOL
        IARRAY_DATA_TYPE_MAX

    ctypedef enum iarray_compression_codec_t:
        IARRAY_COMPRESSION_BLOSCLZ = 0
        IARRAY_COMPRESSION_LZ4
        IARRAY_COMPRESSION_LZ4HC
        IARRAY_COMPRESSION_SNAPPY
        IARRAY_COMPRESSION_ZLIB
        IARRAY_COMPRESSION_ZSTD
        IARRAY_COMPRESSION_ZFP_FIXED_ACCURACY
        IARRAY_COMPRESSION_ZFP_FIXED_PRECISION
        IARRAY_COMPRESSION_ZFP_FIXED_RATE

    ctypedef enum iarray_compression_favor_t:
        IARRAY_COMPRESSION_FAVOR_BALANCE = 0
        IARRAY_COMPRESSION_FAVOR_SPEED
        IARRAY_COMPRESSION_FAVOR_CRATIO

    ctypedef enum iarray_eval_method_t:
        IARRAY_EVAL_METHOD_AUTO
        IARRAY_EVAL_METHOD_ITERCHUNK
        IARRAY_EVAL_METHOD_ITERBLOSC
        IARRAY_EVAL_METHOD_ITERBLOSC2

    ctypedef enum iarray_eval_engine_t:
        IARRAY_EVAL_ENGINE_AUTO
        IARRAY_EVAL_ENGINE_INTERPRETER
        IARRAY_EVAL_ENGINE_COMPILER

    ctypedef enum iarray_split_mode_t:
        IARRAY_ALWAYS_SPLIT = 1
        IARRAY_NEVER_SPLIT = 2
        IARRAY_AUTO_SPLIT = 3
        IARRAY_FORWARD_COMPAT_SPLIT = 4

    ctypedef struct iarray_config_t:
        iarray_compression_codec_t compression_codec
        int compression_level
        iarray_compression_favor_t compression_favor
        int use_dict
        int splitmode
        int filter_flags
        unsigned int eval_method
        int max_num_threads
        uint8_t fp_mantissa_bits
        bool btune
        uint8_t compression_meta

    ctypedef struct iarray_storage_t:
        const char *urlpath
        bool contiguous
        int64_t chunkshape[IARRAY_DIMENSION_MAX]
        int64_t blockshape[IARRAY_DIMENSION_MAX]

    ctypedef struct iarray_dtshape_t:
        iarray_data_type_t dtype
        int32_t dtype_size
        int8_t ndim
        int64_t shape[IARRAY_DIMENSION_MAX]

    ctypedef struct iarray_context_t

    ctypedef struct iarray_container_t:
        iarray_dtshape_t *dtshape
        # iarray_auxshape_t *auxshape;
        # caterva_array_t *catarr;
        iarray_storage_t *storage
        bool view
        bool transposed
        #union { float f; double d; } scalar_value;

    ctypedef struct _iarray_jug_var_t:
        const char *var
        iarray_container_t *c

    ctypedef union iarray_user_param_t:
        float f32
        double f64
        int32_t i32
        int64_t i64
        bool b

    ctypedef struct iarray_expression_t:
        iarray_context_t *ctx
        # ina_str_t expr;
        int32_t typesize
        int64_t nbytes
        int nvars
        int32_t max_out_len
        # jug_expression_t *jug_expr;
        uint64_t jug_expr_func
        iarray_dtshape_t *out_dtshape
        iarray_storage_t *out_store_properties
        iarray_container_t *out
        _iarray_jug_var_t vars[IARRAY_EXPR_OPERANDS_MAX]
        iarray_user_param_t user_params[IARRAY_EXPR_USER_PARAMS_MAX]
        int nuser_params

    ina_rc_t iarray_init()

    void iarray_destroy()

    const char* iarray_err_strerror(ina_rc_t rc)

    ina_rc_t iarray_context_new(iarray_config_t *cfg,
                                iarray_context_t **ctx)

    void iarray_context_free(iarray_context_t **ctx)

    ina_rc_t iarray_get_ncores(int *ncores, int64_t max_ncores);

    ina_rc_t iarray_get_L2_size(uint64_t *l2_size);

    ina_rc_t iarray_partition_advice(iarray_context_t *ctx, iarray_dtshape_t *dtshape,
                                     iarray_storage_t *storage,
                                     int64_t min_chunksize, int64_t max_chunksize,
                                     int64_t min_blocksize, int64_t max_blocksize);

    ina_rc_t iarray_empty(iarray_context_t *ctx,
                                       iarray_dtshape_t *dtshape,
                                       iarray_storage_t *storage,
                                       iarray_container_t **container)

    ina_rc_t iarray_uninit(iarray_context_t *ctx,
                           iarray_dtshape_t *dtshape,
                           iarray_storage_t *storage,
                           iarray_container_t ** container)

    void iarray_container_free(iarray_context_t *ctx,
                               iarray_container_t **container)

    ina_rc_t iarray_container_info(iarray_container_t *c,
                                   int64_t *nbytes,
                                   int64_t *cbytes)

    ina_rc_t iarray_arange(iarray_context_t *ctx,
                                iarray_dtshape_t *dtshape,
                                double start,
                                double step,
                                iarray_storage_t *storage,
                                iarray_container_t **container)

    ina_rc_t iarray_linspace(iarray_context_t *ctx,
                             iarray_dtshape_t *dtshape,
                             double start,
                             double stop,
                             iarray_storage_t *storage,
                             iarray_container_t **container)

    ina_rc_t iarray_zeros(iarray_context_t *ctx,
                          iarray_dtshape_t *dtshape,
                          iarray_storage_t *storage,
                          iarray_container_t **container)

    ina_rc_t iarray_ones(iarray_context_t *ctx,
                         iarray_dtshape_t *dtshape,
                         iarray_storage_t *storage,
                         iarray_container_t **container)

    ina_rc_t iarray_fill(iarray_context_t *ctx,
                         iarray_dtshape_t *dtshape,
                         void *value,
                         iarray_storage_t *storage,
                         iarray_container_t ** container)

    ina_rc_t iarray_copy(iarray_context_t *ctx,
                              iarray_container_t *src,
                              bool view,
                              iarray_storage_t *storage,
                              iarray_container_t **dest) nogil

    ina_rc_t iarray_get_slice(iarray_context_t *ctx,
                                   iarray_container_t *src,
                                   int64_t *start,
                                   int64_t *stop,
                                   bool view,
                                   iarray_storage_t *storage,
                                   iarray_container_t **container)

    ina_rc_t iarray_set_slice_buffer(iarray_context_t *ctx,
                                     iarray_container_t *container,
                                     const int64_t *start,
                                     const int64_t *stop,
                                     void *buffer,
                                     int64_t buflen)

    ina_rc_t iarray_to_buffer(iarray_context_t *ctx,
                                   iarray_container_t *container,
                                   void *buffer,
                                   int64_t buflen)

    ina_rc_t iarray_container_resize(iarray_context_t *ctx,
                                     iarray_container_t *container,
                                     int64_t *new_shape,
                                     int64_t *start)
    ina_rc_t iarray_container_insert(iarray_context_t *ctx,
                                     iarray_container_t *container,
                                     void *buffer,
                                     int64_t buffersize,
                                     const int8_t axis,
                                     int64_t insert_start)
    ina_rc_t iarray_container_append(iarray_context_t *ctx,
                                     iarray_container_t *container,
                                     void *buffer,
                                     int64_t buffersize,
                                     const int8_t axis)
    ina_rc_t iarray_container_delete(iarray_context_t *ctx,
                                     iarray_container_t *container,
                                     const int8_t axis,
                                     int64_t delete_start,
                                     int64_t delete_len)

    ina_rc_t iarray_squeeze_index(iarray_context_t *ctx,
                                  iarray_container_t *container,
                                  bool *index)

    ina_rc_t iarray_squeeze(iarray_context_t *ctx,
                            iarray_container_t *container)

    ina_rc_t iarray_get_dtshape(iarray_context_t *ctx,
                                iarray_container_t *c,
                                iarray_dtshape_t *dtshape)

    ina_rc_t iarray_get_storage(iarray_context_t *ctx,
                                iarray_container_t *c,
                                iarray_storage_t *storage)

    ina_rc_t iarray_get_cfg(iarray_context_t *ctx,
                            iarray_container_t *c,
                            iarray_config_t *cfg)

    ina_rc_t iarray_is_view(iarray_context_t *ctx,
                            iarray_container_t *c,
                            bool *view)

    ina_rc_t iarray_is_transposed(iarray_context_t *ctx,
                                  iarray_container_t *c,
                                  bool *transposed)

    ina_rc_t iarray_from_buffer(iarray_context_t *ctx,
                                     iarray_dtshape_t *dtshape,
                                     void *buffer,
                                     int64_t buflen,
                                     iarray_storage_t *storage,
                                     iarray_container_t **container)

    ina_rc_t iarray_container_load(iarray_context_t *ctx,
                                        char *urlpath,
                                        iarray_container_t **container)

    ina_rc_t iarray_container_open(iarray_context_t *ctx,
                                        char *urlpath,
                                        iarray_container_t **container)

    bool iarray_is_empty(iarray_container_t *container)

    ina_rc_t iarray_expr_new(iarray_context_t *ctx,
                             iarray_data_type_t dtype,
                             iarray_expression_t **e)
    void iarray_expr_free(iarray_context_t *ctx,
                          iarray_expression_t **e)

    ina_rc_t iarray_expr_bind(iarray_expression_t *e, const char *var, iarray_container_t *val)
    ina_rc_t iarray_expr_bind_param(iarray_expression_t *e, iarray_user_param_t val)
    ina_rc_t iarray_expr_bind_out_properties(iarray_expression_t *e,
                                             iarray_dtshape_t *dtshape,
                                             iarray_storage_t *store);
    ina_rc_t iarray_expr_compile(iarray_expression_t *e, const char *expr)

    ina_rc_t iarray_expr_compile_udf(iarray_expression_t *e,
                                     int llvm_bc_len,
                                     const char *llvm_bc,
                                     const char *name)

    ina_rc_t iarray_eval(iarray_expression_t *e, iarray_container_t **c) nogil


    # Linear algebra

    ctypedef enum iarray_operator_hint_t:
        IARRAY_OPERATOR_GENERAL = 0
        IARRAY_OPERATOR_SYMMETRIC
        IARRAY_OPERATOR_TRIANGULAR

    ina_rc_t iarray_matmul_advice(iarray_context_t *ctx,
                                  iarray_container_t *a,
                                  iarray_container_t *b,
                                  iarray_container_t *c,
                                  int64_t *blockshape_a,
                                  int64_t *blockshape_b,
                                  int64_t low,
                                  int64_t high)

    ina_rc_t iarray_linalg_matmul(iarray_context_t *ctx,
                                  iarray_container_t *a,
                                  iarray_container_t *b,
                                  iarray_storage_t *store,
                                  iarray_container_t **result)

    ina_rc_t iarray_opt_gemv(iarray_context_t *ctx,
                                  iarray_container_t *a,
                                  iarray_container_t *b,
                                  iarray_storage_t *store,
                                  iarray_container_t **result)

    ina_rc_t iarray_opt_gemm(iarray_context_t *ctx,
                                  iarray_container_t *a,
                                  iarray_container_t *b,
                                  iarray_storage_t *store,
                                  iarray_container_t **result)

    ina_rc_t iarray_opt_gemm_b(iarray_context_t *ctx,
                                  iarray_container_t *a,
                                  iarray_container_t *b,
                                  iarray_storage_t *store,
                                  iarray_container_t **result)

    ina_rc_t iarray_opt_gemm_a(iarray_context_t *ctx,
                                  iarray_container_t *a,
                                  iarray_container_t *b,
                                  iarray_storage_t *store,
                                  iarray_container_t **result)

    ina_rc_t iarray_linalg_transpose(iarray_context_t *ctx,
                                     iarray_container_t *a,
                                     iarray_container_t **b);

    # Reductions

    ctypedef enum iarray_reduce_func_t:
        IARRAY_REDUCE_MAX,
        IARRAY_REDUCE_MIN,
        IARRAY_REDUCE_SUM,
        IARRAY_REDUCE_PROD,
        IARRAY_REDUCE_MEAN,
        IARRAY_REDUCE_VAR,
        IARRAY_REDUCE_STD,
        IARRAY_REDUCE_MEDIAN

    ina_rc_t iarray_reduce(iarray_context_t *ctx,
                           iarray_container_t *a,
                           iarray_reduce_func_t func,
                           int8_t axis,
                           iarray_storage_t *store,
                           iarray_container_t **b);

    ina_rc_t iarray_reduce_multi(iarray_context_t *ctx,
                                 iarray_container_t *a,
                                 iarray_reduce_func_t func,
                                 int8_t naxis,
                                 int8_t *axis,
                                 iarray_storage_t *store,
                                 iarray_container_t **b);


    # Iterators

    ctypedef struct iarray_iter_write_block_t

    ctypedef struct iarray_iter_write_block_value_t:
        void *block_pointer
        int64_t *block_index
        int64_t *elem_index
        int64_t nblock
        int64_t *block_shape
        int64_t block_size

    ina_rc_t iarray_iter_write_block_new(iarray_context_t *ctx,
                                         iarray_iter_write_block_t **itr,
                                         iarray_container_t *container,
                                         const int64_t *blockshape,
                                         iarray_iter_write_block_value_t *value,
                                         bool external_buffer)
    void iarray_iter_write_block_free(iarray_iter_write_block_t **itr)
    ina_rc_t iarray_iter_write_block_next(iarray_iter_write_block_t *itr, void *buffer, int32_t bufsize)
    ina_rc_t iarray_iter_write_block_has_next(iarray_iter_write_block_t *itr)


    ctypedef struct iarray_iter_read_block_t

    ctypedef struct iarray_iter_read_block_value_t:
        void *block_pointer
        int64_t *block_index
        int64_t *elem_index
        int64_t nblock
        int64_t* block_shape
        int64_t block_size


    ina_rc_t iarray_iter_read_block_new(iarray_context_t *ctx,
                                        iarray_iter_read_block_t **itr,
                                        iarray_container_t *cont,
                                        const int64_t *blockshape,
                                        iarray_iter_read_block_value_t *value,
                                        bool external_buffer)
    void iarray_iter_read_block_free(iarray_iter_read_block_t **itr)
    ina_rc_t iarray_iter_read_block_next(iarray_iter_read_block_t *itr, void *buffer, int32_t bufsize)
    ina_rc_t iarray_iter_read_block_has_next(iarray_iter_read_block_t *itr)

    # Random

    ctypedef enum iarray_random_rng_t:
        IARRAY_RANDOM_RNG_MRG32K3A

    ctypedef enum iarray_random_dist_parameter_t:
        IARRAY_RANDOM_DIST_PARAM_MU
        IARRAY_RANDOM_DIST_PARAM_SIGMA
        IARRAY_RANDOM_DIST_PARAM_ALPHA
        IARRAY_RANDOM_DIST_PARAM_BETA
        IARRAY_RANDOM_DIST_PARAM_LAMBDA
        IARRAY_RANDOM_DIST_PARAM_A
        IARRAY_RANDOM_DIST_PARAM_B
        IARRAY_RANDOM_DIST_PARAM_P
        IARRAY_RANDOM_DIST_PARAM_M
        IARRAY_RANDOM_DIST_PARAM_SENTINEL

    ctypedef struct iarray_random_ctx_t

    ina_rc_t iarray_random_ctx_new(iarray_context_t *ctx,
                                   uint32_t seed,
                                   iarray_random_rng_t rng,
                                   iarray_random_ctx_t **rng_ctx)

    void iarray_random_ctx_free(iarray_context_t *ctx, iarray_random_ctx_t **rng_ctx)

    ina_rc_t iarray_random_dist_set_param(iarray_random_ctx_t *ctx,
                                          iarray_random_dist_parameter_t key,
                                          double value)

    ina_rc_t iarray_random_rand(iarray_context_t *ctx,
                                iarray_dtshape_t *dtshape,
                                iarray_random_ctx_t *rand_ctx,
                                iarray_storage_t *store,
                                iarray_container_t **container)

    ina_rc_t iarray_random_randn(iarray_context_t *ctx,
                                 iarray_dtshape_t *dtshape,
                                 iarray_random_ctx_t *rand_ctx,
                                 iarray_storage_t *store,
                                 iarray_container_t **container)

    ina_rc_t iarray_random_beta(iarray_context_t *ctx,
                                iarray_dtshape_t *dtshape,
                                iarray_random_ctx_t *rand_ctx,
                                iarray_storage_t *store,
                                iarray_container_t **container)

    ina_rc_t iarray_random_lognormal(iarray_context_t *ctx,
                                     iarray_dtshape_t *dtshape,
                                     iarray_random_ctx_t *rand_ctx,
                                     iarray_storage_t *store,
                                     iarray_container_t **container)

    ina_rc_t iarray_random_exponential(iarray_context_t *ctx,
                                       iarray_dtshape_t *dtshape,
                                       iarray_random_ctx_t *random_ctx,
                                       iarray_storage_t *store,
                                       iarray_container_t **container)


    ina_rc_t iarray_random_uniform(iarray_context_t *ctx,
                                   iarray_dtshape_t *dtshape,
                                   iarray_random_ctx_t *random_ctx,
                                   iarray_storage_t *store,
                                   iarray_container_t **container)

    ina_rc_t iarray_random_normal(iarray_context_t *ctx,
                                  iarray_dtshape_t *dtshape,
                                  iarray_random_ctx_t *random_ctx,
                                  iarray_storage_t *store,
                                  iarray_container_t **container)

    ina_rc_t iarray_random_bernoulli(iarray_context_t *ctx,
                                     iarray_dtshape_t *dtshape,
                                     iarray_random_ctx_t *random_ctx,
                                     iarray_storage_t *store,
                                     iarray_container_t **container)

    ina_rc_t iarray_random_binomial(iarray_context_t *ctx,
                                    iarray_dtshape_t *dtshape,
                                    iarray_random_ctx_t *random_ctx,
                                    iarray_storage_t *store,
                                    iarray_container_t **container)

    ina_rc_t iarray_random_poisson(iarray_context_t *ctx,
                                   iarray_dtshape_t *dtshape,
                                   iarray_random_ctx_t *random_ctx,
                                   iarray_storage_t *store,
                                   iarray_container_t **container)

    ina_rc_t iarray_random_kstest(iarray_context_t *ctx,
                                       iarray_container_t *container1,
                                       iarray_container_t *container2,
                                       bool *res)

    # Vlmetalayers

    ctypedef struct iarray_metalayer_t:
        char *name;
        uint8_t *sdata;
        int32_t size;

    ina_rc_t iarray_vlmeta_exists(iarray_context_t *ctx, iarray_container_t *c, const char *name, bool *exists);
    ina_rc_t iarray_vlmeta_add(iarray_context_t *ctx, iarray_container_t *c, iarray_metalayer_t *meta);
    ina_rc_t iarray_vlmeta_update(iarray_context_t *ctx, iarray_container_t *c, iarray_metalayer_t *meta);
    ina_rc_t iarray_vlmeta_get(iarray_context_t *ctx, iarray_container_t *c, const char *name, iarray_metalayer_t *meta);
    ina_rc_t iarray_vlmeta_delete(iarray_context_t *ctx, iarray_container_t *c, const char *name);
    ina_rc_t iarray_vlmeta_nitems(iarray_context_t *ctx, iarray_container_t *c, int16_t *nitems);
    ina_rc_t iarray_vlmeta_get_names(iarray_context_t *ctx, iarray_container_t *c, char ** names);

    # Zarr proxy

    ctypedef void (*zhandler_ptr) (char *zarr_urlpath, int64_t *slice_start, int64_t *slice_stop, uint8_t *dest)
    ina_rc_t iarray_add_zproxy_postfilter(iarray_container_t *src, char *zarr_urlpath, zhandler_ptr zhandler)

    # UDF registry and library functionality
    ctypedef struct iarray_udf_library_t
    ina_rc_t iarray_udf_library_new(const char *name,
                                    iarray_udf_library_t **lib);
    void iarray_udf_library_free(iarray_udf_library_t **lib);
    ina_rc_t iarray_udf_func_register(iarray_udf_library_t *lib,
                                      int llvm_bc_len,
                                      const char *llvm_bc,
                                      iarray_data_type_t return_type,
                                      int num_args,
                                      iarray_data_type_t *arg_types,
                                      const char *name);
    ina_rc_t iarray_udf_func_lookup(const char *full_name,
                                    uint64_t *function_ptr);

    # Indexing

    ina_rc_t iarray_set_orthogonal_selection(iarray_context_t *ctx,
                                    iarray_container_t *c,
                                    int64_t ** selection, int64_t *selection_size,
                                    void *buffer,
                                    int64_t *buffer_shape,
                                    int64_t buffer_size);

    ina_rc_t iarray_get_orthogonal_selection(iarray_context_t *ctx,
                                    iarray_container_t *c,
                                    int64_t ** selection, int64_t *selection_size,
                                    void *buffer,
                                    int64_t *buffer_shape,
                                    int64_t buffer_size);
