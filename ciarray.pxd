cdef extern from "<stdint.h>":
    ctypedef unsigned long long uint64_t

cdef extern from "libiarray/iarray.h":
    ctypedef uint64_t ina_rc_t
    ina_rc_t iarray_init()
    void iarray_destroy()

