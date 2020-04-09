cmake_minimum_required(VERSION 3.11)
project(iarray VERSION 0.1.0)

if(NOT SKBUILD)
  message(FATAL_ERROR "This project is only meant to be built with scikit-build")
endif()

# The inac dependency
if(NOT EXISTS "${CMAKE_BINARY_DIR}/inac.cmake")
    if (NOT EXISTS "${CMAKE_SOURCE_DIR}/inac.cmake")
        message(STATUS "Downloading inac.cmake from https://github.com/inaos/inac-cmake")
        file(DOWNLOAD "https://raw.githubusercontent.com/inaos/inac-cmake/0.3/inac.cmake"
                "${CMAKE_BINARY_DIR}/inac.cmake" STATUS DS)
        if(NOT "${DS}"  MATCHES "0;")
            file(REMOVE "${CMAKE_BINARY_DIR}/inac.cmake")
            message(FATAL_ERROR "Failed to download inac.cmake")
        endif()
    else()
        message(STATUS "Use local inac.cmake")
        configure_file("${CMAKE_SOURCE_DIR}/inac.cmake" "${CMAKE_BINARY_DIR}/inac.cmake" COPYONLY)
    endif()
endif()
include("${CMAKE_BINARY_DIR}/inac.cmake")
inac_add_dependency(inac "1.0.7")

# The Python wrapper is the iarray directory
add_subdirectory(iarray)
