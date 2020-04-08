file(WRITE version.py "version = '${CMAKE_PROJECT_VERSION}'\n")

add_cython_target(iarray_ext "iarray_ext.pyx")
add_library(iarray_ext MODULE ${iarray_ext})
target_link_libraries(iarray_ext "iarray")
python_extension_module(iarray_ext)

add_custom_command(TARGET iarray_ext POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:iarray_ext> ${CMAKE_SOURCE_DIR}/iarray
)

install(TARGETS iarray_ext LIBRARY DESTINATION iarray)
