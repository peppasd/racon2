#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "spoa::spoa" for configuration ""
set_property(TARGET spoa::spoa APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(spoa::spoa PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libspoa.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS spoa::spoa )
list(APPEND _IMPORT_CHECK_FILES_FOR_spoa::spoa "${_IMPORT_PREFIX}/lib/libspoa.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
