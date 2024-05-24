add_library(polybench_utils STATIC polybench.cpp)
target_include_directories(polybench_utils PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(PB_KOKKOS)
  target_link_libraries(polybench_utils PUBLIC kokkos)
endif()
