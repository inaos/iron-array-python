cdef extern from "<stdint.h>":
    ctypedef unsigned long long uint64_t

cdef extern from "/Users/aleix11alcacer/Documents/Francesc Alted/IronArray/iron-array/include/libiarray/iarray.h":
    ctypedef uint64_t ina_rc_t
    extern ina_rc_t iarray_init()
    extern void iarray_destroy()
