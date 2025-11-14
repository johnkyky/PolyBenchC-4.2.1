add_executable(trmm trmm.cpp)
target_include_directories(trmm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(trmm PRIVATE m polybench_utils)
