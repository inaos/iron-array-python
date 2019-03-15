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

    cdef enum:
        IARRAY_DIMENSION_MAX = 8

    ctypedef enum iarray_container_flags_t:
        IARRAY_CONTAINER_PERSIST = 0x1

    ctypedef enum iarray_data_type_t:
        IARRAY_DATA_TYPE_DOUBLE,
        IARRAY_DATA_TYPE_FLOAT,
        IARRAY_DATA_TYPE_MAX

    ctypedef enum iarray_compression_codec_t:
        IARRAY_COMPRESSION_BLOSCLZ = 0,
        IARRAY_COMPRESSION_LZ4,
        IARRAY_COMPRESSION_LZ4HC,
        IARRAY_COMPRESSION_SNAPPY,
        IARRAY_COMPRESSION_ZLIB,
        IARRAY_COMPRESSION_ZSTD,
        IARRAY_COMPRESSION_LIZARD

    ctypedef enum iarray_eval_flags_t:
        IARRAY_EXPR_EVAL_BLOCK = 0x1
        IARRAY_EXPR_EVAL_CHUNK = 0x2
        IARRAY_EXPR_EVAL_ITERBLOCK = 0x4
        IARRAY_EXPR_EVAL_ITERCHUNK = 0x8

    ctypedef struct iarray_config_t:
        iarray_compression_codec_t compression_codec
        int compression_level
        int filter_flags
        int eval_flags
        int max_num_threads
        int8_t fp_mantissa_bits
        int blocksize

    ctypedef struct iarray_store_properties_t:
        const char *id

    ctypedef struct iarray_dtshape_t:
        iarray_data_type_t dtype
        int8_t ndim
        int64_t shape[IARRAY_DIMENSION_MAX]
        int64_t pshape[IARRAY_DIMENSION_MAX]

    ctypedef struct iarray_context_t

    ctypedef struct iarray_container_t

    ctypedef struct iarray_expression_t

    ina_rc_t iarray_init()

    void iarray_destroy()

    ina_rc_t iarray_context_new(iarray_config_t *cfg,
                                iarray_context_t **ctx)

    void iarray_context_free(iarray_context_t **ctx)

    ina_rc_t iarray_container_new(iarray_context_t *ctx,
                                  iarray_dtshape_t *dtshape,
                                  iarray_store_properties_t *store,
                                  int flags,
                                  iarray_container_t **container)

    void iarray_container_free(iarray_context_t *ctx,
                               iarray_container_t **container)

    ina_rc_t iarray_arange(iarray_context_t *ctx,
                           iarray_dtshape_t *dtshape,
                           double start,
                           double stop,
                           double step,
                           iarray_store_properties_t *store,
                           int flags,
                           iarray_container_t **container)

    ina_rc_t iarray_linspace(iarray_context_t *ctx,
                             iarray_dtshape_t *dtshape,
                             int64_t nelem,
                             double start,
                             double stop,
                             iarray_store_properties_t *store,
                             int flags,
                             iarray_container_t **container)

    ina_rc_t iarray_zeros(iarray_context_t *ctx,
                          iarray_dtshape_t *dtshape,
                          iarray_store_properties_t *store,
                          int flags,
                          iarray_container_t **container)

    ina_rc_t iarray_ones(iarray_context_t *ctx,
                         iarray_dtshape_t *dtshape,
                         iarray_store_properties_t *store,
                         int flags,
                         iarray_container_t **container)

    ina_rc_t iarray_fill_float(iarray_context_t *ctx,
                               iarray_dtshape_t *dtshape,
                               float value,
                               iarray_store_properties_t *store,
                               int flags,
                               iarray_container_t **container)

    ina_rc_t iarray_fill_double(iarray_context_t *ctx,
                                     iarray_dtshape_t *dtshape,
                                     double value,
                                     iarray_store_properties_t *store,
                                     int flags,
                                     iarray_container_t **container)

    ina_rc_t iarray_get_slice(iarray_context_t *ctx,
                              iarray_container_t *c,
                              int64_t *start,
                              int64_t *stop,
                              int64_t *pshape,
                              iarray_store_properties_t *store,
                              int flags,
                              bool view,
                              iarray_container_t **container)

    ina_rc_t iarray_to_buffer(iarray_context_t *ctx,
                              iarray_container_t *container,
                              void *buffer,
                              size_t buffer_len)

    ina_rc_t iarray_get_dtshape(iarray_context_t *ctx,
                                iarray_container_t *c,
                                iarray_dtshape_t *dtshape)

    ina_rc_t iarray_from_buffer(iarray_context_t *ctx,
                                iarray_dtshape_t *dtshape,
                                void *buffer,
                                size_t buffer_len,
                                iarray_store_properties_t *store,
                                int flags,
                                iarray_container_t **container)

    ina_rc_t iarray_from_file(iarray_context_t *ctx,
                              iarray_store_properties_t *store,
                              iarray_container_t **container)

    ina_rc_t iarray_expr_new(iarray_context_t *ctx, iarray_expression_t **e)
    void iarray_expr_free(iarray_context_t *ctx, iarray_expression_t **e)

    ina_rc_t iarray_expr_bind(iarray_expression_t *e, const char *var, iarray_container_t *val)

    ina_rc_t iarray_expr_compile(iarray_expression_t *e, const char *expr)

    ina_rc_t iarray_eval(iarray_expression_t *e, iarray_container_t *ret)

    # Iterators

    ctypedef struct  iarray_iter_write_part_t

    ctypedef struct iarray_iter_write_part_value_t:
        void *pointer
        int64_t *part_index
        int64_t *elem_index
        int64_t nelem
        int64_t* part_shape

    ina_rc_t iarray_iter_write_part_new(iarray_context_t *ctx, iarray_container_t *container,
                                             iarray_iter_write_part_t **itr)
    void iarray_iter_write_part_free(iarray_iter_write_part_t *itr)
    void iarray_iter_write_part_init(iarray_iter_write_part_t *itr)
    ina_rc_t iarray_iter_write_part_next(iarray_iter_write_part_t *itr)
    int iarray_iter_write_part_finished(iarray_iter_write_part_t *itr)
    void iarray_iter_write_part_value(iarray_iter_write_part_t *itr, iarray_iter_write_part_value_t *value)

    ctypedef struct iarray_iter_read_t

    ctypedef struct iarray_iter_read_value_t:
        void *pointer
        int64_t *index
        int64_t nelem

    ina_rc_t iarray_iter_read_new(iarray_context_t *ctx, iarray_container_t *container,
                                       iarray_iter_read_t **itr);
    void iarray_iter_read_free(iarray_iter_read_t *itr);
    void iarray_iter_read_init(iarray_iter_read_t *itr);
    ina_rc_t iarray_iter_read_next(iarray_iter_read_t *itr);
    int iarray_iter_read_finished(iarray_iter_read_t *itr);
    void iarray_iter_read_value(iarray_iter_read_t *itr, iarray_iter_read_value_t *val);

    ctypedef struct iarray_iter_read_block_t

    ctypedef struct iarray_iter_read_block_value_t:
        void *pointer
        int64_t *block_index
        int64_t *elem_index
        int64_t nelem
        int64_t* block_shape

    ina_rc_t iarray_iter_read_block_new(iarray_context_t *ctx, iarray_container_t *container,
                                                 iarray_iter_read_block_t **itr, int64_t *blockshape)
    void iarray_iter_read_block_free(iarray_iter_read_block_t *itr)
    void iarray_iter_read_block_init(iarray_iter_read_block_t *itr)
    ina_rc_t iarray_iter_read_block_next(iarray_iter_read_block_t *itr)
    int iarray_iter_read_block_finished(iarray_iter_read_block_t *itr)
    void iarray_iter_read_block_value(iarray_iter_read_block_t *itr, iarray_iter_read_block_value_t *value)

    # Random

    ctypedef enum iarray_random_rng_t:
        IARRAY_RANDOM_RNG_MERSENNE_TWISTER
        IARRAY_RANDOM_RNG_SOBOL

    ctypedef enum iarray_random_dist_parameter_t:
        IARRAY_RANDOM_DIST_PARAM_MU
        IARRAY_RANDOM_DIST_PARAM_SIGMA
        IARRAY_RANDOM_DIST_PARAM_ALPHA
        IARRAY_RANDOM_DIST_PARAM_BETA
        IARRAY_RANDOM_DIST_PARAM_A
        IARRAY_RANDOM_DIST_PARAM_B
        IARRAY_RANDOM_DIST_PARAM_SENTINEL

    ctypedef struct iarray_random_ctx_t

    ina_rc_t iarray_random_ctx_new(iarray_context_t *ctx,
                                   uint32_t seed,
                                   iarray_random_rng_t rng,
                                   iarray_random_ctx_t **rng_ctx)

    void iarray_random_ctx_free(iarray_context_t *ctx, iarray_random_ctx_t **rng_ctx)

    ina_rc_t iarray_random_dist_set_param_float(iarray_random_ctx_t *ctx,
                                                iarray_random_dist_parameter_t key,
                                                float value)

    ina_rc_t iarray_random_dist_set_param_double(iarray_random_ctx_t *ctx,
                                                 iarray_random_dist_parameter_t key,
                                                 double value)

    ina_rc_t iarray_random_rand(iarray_context_t *ctx,
                                iarray_dtshape_t *dtshape,
                                iarray_random_ctx_t *rand_ctx,
                                iarray_store_properties_t *store,
                                int flags,
                                iarray_container_t **container)

    ina_rc_t iarray_random_randn(iarray_context_t *ctx,
                                 iarray_dtshape_t *dtshape,
                                 iarray_random_ctx_t *rand_ctx,
                                 iarray_store_properties_t *store,
                                 int flags,
                                 iarray_container_t **container)

    ina_rc_t iarray_random_beta(iarray_context_t *ctx,
                                iarray_dtshape_t *dtshape,
                                iarray_random_ctx_t *rand_ctx,
                                iarray_store_properties_t *store,
                                int flags,
                                iarray_container_t **container)

    ina_rc_t iarray_random_lognormal(iarray_context_t *ctx,
                                     iarray_dtshape_t *dtshape,
                                     iarray_random_ctx_t *rand_ctx,
                                     iarray_store_properties_t *store,
                                     int flags,
                                     iarray_container_t **container)

    ina_rc_t iarray_random_exponential(iarray_context_t *ctx,
                                       iarray_dtshape_t *dtshape,
                                       iarray_random_ctx_t *random_ctx,
                                       iarray_store_properties_t *store,
                                       int flags,
                                       iarray_container_t **container)


    ina_rc_t iarray_random_uniform(iarray_context_t *ctx,
                                   iarray_dtshape_t *dtshape,
                                   iarray_random_ctx_t *random_ctx,
                                   iarray_store_properties_t *store,
                                   int flags,
                                   iarray_container_t **container)

    ina_rc_t iarray_random_kstest(iarray_context_t *ctx,
                                  iarray_container_t *c1,
                                  iarray_container_t *c2,
                                  bool *res)