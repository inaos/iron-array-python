cmake_minimum_required(VERSION 3.11)
project(iarray VERSION 0.1.0)

if(NOT SKBUILD)
  message(FATAL_ERROR "This project is only meant to be built with scikit-build")
endif()

# The Python wrapper
add_subdirectory(iarray)
