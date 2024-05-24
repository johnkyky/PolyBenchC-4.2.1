add_executable(jacobi-1d jacobi-1d.cpp)
target_include_directories(jacobi-1d PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(jacobi-1d PRIVATE m polybench_utils)
