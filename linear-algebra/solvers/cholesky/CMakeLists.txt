add_executable(cholesky cholesky.cpp)
target_include_directories(cholesky PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cholesky PRIVATE m polybench_utils)
