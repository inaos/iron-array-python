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

# Write the version into a python file
file(WRITE version.py "version = '${CMAKE_PROJECT_VERSION}'\n")

# Find dependencies
find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)
find_package(NumPy REQUIRED)
include_directories(${NumPy_INCLUDE_DIRS})
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/iarray-c-develop/include")
link_directories("${CMAKE_CURRENT_SOURCE_DIR}/iarray-c-develop/build")

# Declare the target for the iarray extension
add_cython_target(iarray_ext "iarray_ext.pyx")
add_library(iarray_ext MODULE ${iarray_ext})

# Build the C library (only the shared one is necessary) *before* the extension
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/iarray-c-develop/build")
add_custom_command(
    TARGET iarray_ext PRE_BUILD
    COMMAND mkdir build
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/iarray-c-develop"
)
endif(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/iarray-c-develop/build")
add_custom_command(
    TARGET iarray_ext PRE_BUILD
    COMMAND ${CMAKE_COMMAND} .. -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/iarray-c-develop/build"
)
add_custom_command(
    TARGET iarray_ext PRE_BUILD
    COMMAND ${CMAKE_COMMAND} --build . --target iarray -- -j 4
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/iarray-c-develop/build"
)

# Now, build the extension itself
target_link_libraries(iarray_ext "iarray")
python_extension_module(iarray_ext)

# Copy the extension to the iarray directory
add_custom_command(
    TARGET iarray_ext POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:iarray_ext> ${CMAKE_SOURCE_DIR}/iarray
)

install(TARGETS iarray_ext LIBRARY DESTINATION iarray)
