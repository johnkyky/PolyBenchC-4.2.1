add_executable(correlation correlation.cpp)
target_include_directories(correlation PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(correlation PRIVATE m polybench_utils)
