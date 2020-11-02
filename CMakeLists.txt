cmake_minimum_required(VERSION 3.11)
project(iarray VERSION 0.1.0)

option(DISABLE_LLVM_CONFIG "Disable the use of llvm-config for finding libraries" OFF)

if(NOT SKBUILD)
  message(FATAL_ERROR "This project is only meant to be built with scikit-build")
endif()

# If the build type is not set or not a valid value, default to RelWithDebInfo
set(IARRAY_DEFAULT_BUILD_TYPE RelWithDebInfo)
if (NOT CMAKE_BUILD_TYPE OR NOT (CMAKE_BUILD_TYPE STREQUAL Debug OR CMAKE_BUILD_TYPE STREQUAL RelWithDebInfo))
    message(STATUS "No valid build type specified. Defaulting to '${IARRAY_DEFAULT_BUILD_TYPE}'.")
    set(CMAKE_BUILD_TYPE ${IARRAY_DEFAULT_BUILD_TYPE} CACHE STRING
            "Choose the type of build." FORCE)

    # Set the possible values of build type for cmake-gui
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "RelWithDebInfo")
endif ()

# The Python wrapper is the iarray directory
add_subdirectory(iarray)
