# distutils: include_dirs = /Users/aleix11alcacer/INAOS/inac-darwin-x86_64-debug-1.0.1/include

cimport ciarray

def iarray_init():
    return ciarray.iarray_init()

def iarray_destroy():
    return ciarray.iarray_destroy()
