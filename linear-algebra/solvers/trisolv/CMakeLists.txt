add_executable(trisolv trisolv.cpp)
target_include_directories(trisolv PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(trisolv PRIVATE m polybench_utils)
