add_executable(jacobi-2d jacobi-2d.cpp)
target_include_directories(jacobi-2d PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(jacobi-2d PRIVATE m polybench_utils)
