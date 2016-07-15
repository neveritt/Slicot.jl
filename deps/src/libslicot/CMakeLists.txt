# cmake version required
cmake_minimum_required (VERSION 2.6)

# project related information
project (Slicot Fortran)

set (ExtLibs ${BlasLibrary})

if (NOT BlasLibrary STREQUAL ${LapackLibrary})
  list (APPEND ExtLibs ${LapackLibrary})
endif ()

message(STATUS "ExtLibs: ${ExtLibs}")

set (CMAKE_Fortran_FLAGS_RELEASE "-O3 -fPIC")
set (CMAKE_Fortran_FLAGS_DEBUG "-O0 -g -fPIC")

if (CMAKE_BUILD_TYPE STREQUAL "Release")
  message (STATUS "Building for Release")
elseif (CMAKE_BUILD_TYPE STREQUAL "Debug")
  message (STATUS "Building for Debug")
else ()
  message (STATUS "Building for Release")
  set (CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Build type: Debug [Release]" FORCE)
endif ()

add_subdirectory (src)
add_subdirectory (src_aux)
