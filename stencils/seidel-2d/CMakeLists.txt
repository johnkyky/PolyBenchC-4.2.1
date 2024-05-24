add_executable(seidel-2d seidel-2d.cpp)
target_include_directories(seidel-2d PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(seidel-2d PRIVATE m polybench_utils)
