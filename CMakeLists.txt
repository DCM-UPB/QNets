cmake_minimum_required (VERSION 3.5)
include(FindPackageHandleStandardArgs)

project (ffnn LANGUAGES CXX VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${USER_CXX_FLAGS}")

if (USE_COVERAGE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
endif()

# find packages
message(STATUS "Configured GSL_ROOT_DIR: ${GSL_ROOT_DIR}")

if (USE_OPENMP)
  find_package(OpenMP)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} -DOPENMP=1")
  message(STATUS "OPENMP_LIBRARY_PATHS: ${OpenMP_CXX_LIBRARY}")
  message(STATUS "OPENMP_LIBRARIES: ${OpenMP_CXX_LIBRARIES}")
endif()

find_package(GSL)
message(STATUS "GSL_INCLUDE_DIRS: ${GSL_INCLUDE_DIRS}")
message(STATUS "GSL_LIBRARIES: ${GSL_LIBRARIES}")

message(STATUS "Configured CMAKE_CXX_COMPILER: ${CMAKE_CXX_COMPILER}")
message(STATUS "Configured CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")

# set header / library paths
include_directories(include/ "${GSL_INCLUDE_DIRS}") # headers

enable_testing()

# continue with subdirectories
add_subdirectory(src) 
add_subdirectory(test)
add_subdirectory(benchmark)
add_subdirectory(examples)
