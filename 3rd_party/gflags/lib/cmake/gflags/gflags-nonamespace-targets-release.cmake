#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "gflags_shared" for configuration "Release"
set_property(TARGET gflags_shared APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(gflags_shared PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libgflags.so.2.2.2"
  IMPORTED_SONAME_RELEASE "libgflags.so.2.2"
  )

list(APPEND _cmake_import_check_targets gflags_shared )
list(APPEND _cmake_import_check_files_for_gflags_shared "${_IMPORT_PREFIX}/lib/libgflags.so.2.2.2" )

# Import target "gflags_static" for configuration "Release"
set_property(TARGET gflags_static APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(gflags_static PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libgflags.a"
  )

list(APPEND _cmake_import_check_targets gflags_static )
list(APPEND _cmake_import_check_files_for_gflags_static "${_IMPORT_PREFIX}/lib/libgflags.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
