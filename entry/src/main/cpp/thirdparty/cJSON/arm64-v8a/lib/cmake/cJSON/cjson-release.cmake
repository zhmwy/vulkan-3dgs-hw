#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "cjson" for configuration "Release"
set_property(TARGET cjson APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(cjson PROPERTIES
  IMPORTED_LINK_INTERFACE_LIBRARIES_RELEASE "m"
  IMPORTED_LOCATION_RELEASE "/home/muliyanpo/NJU/Projects/tpc_c_cplusplus/lycium/usr/cJSON/arm64-v8a/lib/libcjson.so.1.7.15"
  IMPORTED_SONAME_RELEASE "libcjson.so.1"
  )

list(APPEND _IMPORT_CHECK_TARGETS cjson )
list(APPEND _IMPORT_CHECK_FILES_FOR_cjson "/home/muliyanpo/NJU/Projects/tpc_c_cplusplus/lycium/usr/cJSON/arm64-v8a/lib/libcjson.so.1.7.15" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
