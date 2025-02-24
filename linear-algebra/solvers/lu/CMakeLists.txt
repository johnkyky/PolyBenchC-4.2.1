add_executable(lu lu.cpp)
target_include_directories(lu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(lu PRIVATE m polybench_utils)
