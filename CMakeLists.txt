cmake_minimum_required(VERSION 3.22)

project(PolyBench LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(PB_TIME_MONITORING "Measure the execution time of the kernels" OFF)
option(PB_CYCLE_MONITORING "Measure the number of cycles of the kernels" OFF)
option(PB_NO_FLUSH_CACHE "Disable cache flushes" OFF)
option(PB_LINUX_FIFO_SCHEDULER "Use the Linux FIFO scheduler" OFF)
# cache size
option(PB_STACK_ARRAYS "Use stack allocation arrays" OFF)
option(PB_DUMP_ARRAYS "Dump the arrays on stderr" OFF)
option(PB_PAPI "Use PAPI for performance timing" OFF)
option(PB_USE_C99 "Use standard C99 prototype for the functions" OFF)
option(PB_USE_SCALAR_LB "Use scalar loop bounds instead of parametric ones" OFF)
set(PB_DATASET_SIZE
    "MEDIUM"
    CACHE STRING "Set the size of the dataset")
option(PB_KOKKOS "Use Kokkos" OFF)
set(PB_KOKKOS_DIR
    ""
    CACHE STRING "Set the install directory of Kokkos")
option(PB_USE_POLLY "Use polly" OFF)

if(PB_TIME_MONITORING)
  add_definitions(-DPOLYBENCH_TIME)
  message(STATUS "Time monitoring enabled")
endif()
if(PB_CYCLE_MONITORING)
  add_definitions(-DPOLYBENCH_CYCLE_ACCURATE_TIMER)
  message(STATUS "Cycle monitoring enabled")
endif()
if(PB_NO_FLUSH_CACHE)
  add_definitions(-DPOLYBENCH_NO_FLUSH_CACHE)
  message(STATUS "Cache flushes disabled")
endif()
if(PB_LINUX_FIFO_SCHEDULER)
  add_definitions(-DPOLYBENCH_LINUX_FIFO_SCHEDULER)
  link_libraries(-lc)
  message(STATUS "Linux FIFO scheduler enabled")
endif()
if(PB_STACK_ARRAYS)
  add_definitions(-DPOLYBENCH_STACK_ARRAYS)
  message(STATUS "Stack arrays enabled")
endif()
if(PB_DUMP_ARRAYS)
  add_definitions(-DPOLYBENCH_DUMP_ARRAYS)
  message(STATUS "Dump arrays enabled")
endif()
if(PB_PAPI)
  set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
  # a teseter
  find_package(PAPI REQUIRED)
  add_definitions(-DPOLYBENCH_PAPI)
  link_libraries(${PAPI_LIBRARIES})
  message(STATUS "PAPI profiling enabled")
endif()
if(PB_USE_C99)
  add_definitions(-DPOLYBENCH_USE_C99_PROTO)
  message(STATUS "C99 prototype enabled")
endif()
if(PB_USE_SCALAR_LB)
  add_definitions(-DPOLYBENCH_USE_SCALAR_LB)
  message(STATUS "Scalar loop bounds enabled")
endif()

string(TOUPPER ${PB_DATASET_SIZE} PB_DATASET_SIZE)
if(PB_DATASET_SIZE STREQUAL "MINI")
  add_definitions(-DMINI_DATASET)
  message(STATUS "Mini dataset")
elseif(PB_DATASET_SIZE STREQUAL "SMALL")
  add_definitions(-DSMALL_DATASET)
  message(STATUS "Small dataset")
elseif(PB_DATASET_SIZE STREQUAL "MEDIUM")
  add_definitions(-DMEDIUM_DATASET)
  message(STATUS "Medium dataset")
elseif(PB_DATASET_SIZE STREQUAL "LARGE")
  add_definitions(-DLARGE_DATASET)
  message(STATUS "Large dataset")
elseif(PB_DATASET_SIZE STREQUAL "EXTRALARGE")
  add_definitions(-DEXTRALARGE_DATASET)
  message(STATUS "Extra large dataset")
else()
  message(
    FATAL_ERROR
      "PB_DATASET_SIZE must be one of: MINI, SMALL, MEDIUM, LARGE, EXTRA_LARGE")
endif()

if(PB_KOKKOS OR (NOT PB_KOKKOS_DIR STREQUAL ""))
  if(PB_KOKKOS_DIR STREQUAL "")
    include(FetchContent)
    FetchContent_Declare(
      kokkos
      GIT_REPOSITORY https://github.com/kokkos/kokkos.git
      GIT_TAG 4.1.00)
    FetchContent_MakeAvailable(kokkos)
    message(STATUS "Kokkos enabled from source")
  else()
    find_package(Kokkos REQUIRED PATHS ${PB_KOKKOS_DIR})
    message(STATUS "Kokkos enabled from ${PB_KOKKOS_DIR}")
  endif()
  add_definitions(-DPOLYBENCH_KOKKOS)
else()
  message(STATUS "Kokkos disabled")
endif()

if((PB_KOKKOS OR PB_KOKKOS_DIR) AND PB_USE_POLLY)
  add_definitions(-DPOLYBENCH_USE_POLLY)
  message(STATUS "Polly enabled")

  add_compile_options(
    "SHELL:-mllvm -polly"
    "SHELL:-mllvm -polly-manual-scop-detection"
    "SHELL:-mllvm -polly-process-unprofitable"
    "SHELL:-mllvm -polly-stmt-granularity=store"
    "SHELL:-mllvm -polly-export"
    "SHELL:-mllvm -polly-dot"
    # "SHELL:-mllvm -polly-opt-outer-coincidence=yes"
    # "SHELL:-mllvm -polly-parallel-force"
    "SHELL:-mllvm -polly-optimizer=pluto"
    "SHELL:-mllvm -polly-pluto-path=pluto"
    "SHELL:-mllvm -polly-dependences-computeout=0"
    "-mllvm" "-debug-only=polly-pluto,polly-check-parallelism,polly-extract-annotations,polly-user-assumptions,polly-array-fusion,polly-scops,polly-detect,polly-openscop-exporter"
    # "SHELL:-mllvm -polly-kokkos-check-parallelism=false"
    "SHELL:-mllvm -polly-codegen-emit-rtc-print"
  )
endif()

add_subdirectory(utilities)
add_subdirectory(datamining)
add_subdirectory(linear-algebra)
add_subdirectory(medley)
add_subdirectory(stencils)
