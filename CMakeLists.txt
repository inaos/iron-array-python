cmake_minimum_required(VERSION 3.11.0)
project(iarray-python VERSION 0.1.0)

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

inac_add_dependency(inac "1.0.6")
inac_add_dependency(iarray "0.1.3" SEMANTIC true)

find_package(PythonExtensions REQUIRED)
find_package(Cython)
find_package(NumPy)

include_directories(${NumPy_INCLUDE_DIRS})

add_cython_target(iarray_ext.pyx)

add_library(iarray_ext MODULE ${iarray_ext})
target_link_libraries(iarray_ext ${INAC_DEPENDENCY_LIBS})

python_extension_module(iarray_ext)
install(TARGETS iarray_ext LIBRARY DESTINATION iarray)
