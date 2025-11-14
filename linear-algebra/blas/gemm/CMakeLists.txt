add_executable(gemm gemm.cpp)
target_include_directories(gemm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gemm PRIVATE m polybench_utils)
