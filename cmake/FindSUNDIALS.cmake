# - Find SUNDIALS

# Look for the header file
FIND_PATH(SUNDIALS_INCLUDE_DIR NAMES sundials/sundials_types.h)
# Look for the library
FIND_LIBRARY(SUNDIALS_CVODE_LIBRARY NAMES sundials_cvode)
FIND_LIBRARY(SUNDIALS_NVECS_LIBRARY NAMES sundials_nvecserial)

# handle the QUIETLY and REQUIRED arguments and set SUNDIALS_FOUND to TRUE if
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(SUNDIALS DEFAULT_MSG SUNDIALS_CVODE_LIBRARY SUNDIALS_NVECS_LIBRARY SUNDIALS_INCLUDE_DIR)

# Copy the results to the output variables.
IF(SUNDIALS_FOUND)
  SET(SUNDIALS_LIBRARIES ${SUNDIALS_CVODE_LIBRARY} ${SUNDIALS_NVECS_LIBRARY})
  SET(SUNDIALS_INCLUDE_DIRS ${SUNDIALS_INCLUDE_DIR})
ELSE()
  SET(SUNDIALS_LIBRARIES)
  SET(SUNDIALS_INCLUDE_DIRS)
ENDIF()

MARK_AS_ADVANCED(SUNDIALS_INCLUDE_DIR SUNDIALS_NVECS_LIBRARY SUNDIALS_CVODE_LIBRARY)

if(SUNDIALS_INCLUDE_DIR AND SUNDIALS_LIBRARIES)
  add_library(SUNDIALS::CVODE UNKNOWN IMPORTED)
  set_target_properties(SUNDIALS::CVODE PROPERTIES
    IMPORTED_LOCATION "${SUNDIALS_CVODE_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${SUNDIALS_INCLUDE_DIR}")
  add_library(SUNDIALS::NVECS UNKNOWN IMPORTED)
  set_target_properties(SUNDIALS::NVECS PROPERTIES
    IMPORTED_LOCATION "${SUNDIALS_NVECS_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${SUNDIALS_INCLUDE_DIR}")
endif()