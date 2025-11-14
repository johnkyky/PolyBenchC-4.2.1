add_executable(syr2k syr2k.cpp)
target_include_directories(syr2k PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(syr2k PRIVATE m polybench_utils)
