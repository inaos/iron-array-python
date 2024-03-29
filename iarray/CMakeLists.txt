# Write the version into a python file
file(WRITE version.py "version = '${CMAKE_PROJECT_VERSION}'\n")


# Find dependencies
find_package(PythonExtensions REQUIRED)
find_package(Cython REQUIRED)
find_package(NumPy REQUIRED)
include_directories(${NumPy_INCLUDE_DIRS})
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/iarray-c-develop/include")
link_directories("${CMAKE_CURRENT_SOURCE_DIR}/iarray-c-develop/build")  # needed by windows
link_directories("${CMAKE_CURRENT_SOURCE_DIR}")

if (WIN32)
    add_definitions(-DINA_OS_WINDOWS_SKIP_POSIX_TIME_DEFS)
endif (WIN32)

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

if (WIN32)
    add_custom_command(
        TARGET iarray_ext PRE_BUILD
        # COMMAND ${CMAKE_COMMAND} .. -G ${WINGEN} -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=${CMAKE_CURRENT_SOURCE_DIR} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DINAC_TARGET_ARCH=${INAC_TARGET_ARCH} -DLLVM_DIR=${LLVM_DIR}
        COMMAND ${CMAKE_COMMAND} .. -G ${WINGEN} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DINAC_TARGET_ARCH=${INAC_TARGET_ARCH} -DLLVM_DIR=${LLVM_DIR}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/iarray-c-develop/build"
        )
    add_custom_command(
        TARGET iarray_ext PRE_BUILD
        COMMAND ${CMAKE_COMMAND} --build . --target iarray
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/iarray-c-develop/build"
        )
else(WIN32)
    add_custom_command(
        TARGET iarray_ext PRE_BUILD
        COMMAND ${CMAKE_COMMAND} .. -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=${CMAKE_CURRENT_SOURCE_DIR} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DDISABLE_LLVM_CONFIG=${DISABLE_LLVM_CONFIG} -DLLVM_DIR=${LLVM_DIR}
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/iarray-c-develop/build"
        )
    add_custom_command(
        TARGET iarray_ext PRE_BUILD
        COMMAND ${CMAKE_COMMAND} --build . --target iarray -- -j 4
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/iarray-c-develop/build"
        )
endif(WIN32)


## The Python wrapper is the iarray directory
#include_directories("${CMAKE_CURRENT_SOURCE_DIR}/iarray-c-develop/include")
##link_directories("${CMAKE_CURRENT_SOURCE_DIR}/iarray-c-develop/build")
#link_directories("${CMAKE_CURRENT_SOURCE_DIR}")
#SET(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
#set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/iarray-c-develop")
#add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/iarray-c-develop")

# Now, build the extension itself
target_link_libraries(iarray_ext "iarray")
python_extension_module(iarray_ext)


# Copy the extension to the iarray directory
add_custom_command(
    TARGET iarray_ext POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:iarray_ext> ${CMAKE_SOURCE_DIR}/iarray
    )

install(TARGETS iarray_ext LIBRARY DESTINATION iarray)
