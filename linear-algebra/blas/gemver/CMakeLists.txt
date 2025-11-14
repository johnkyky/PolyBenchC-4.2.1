add_executable(gemver gemver.cpp)
target_include_directories(gemver PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gemver PRIVATE m polybench_utils)
