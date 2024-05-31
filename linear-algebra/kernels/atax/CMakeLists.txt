add_executable(atax atax.cpp)
target_include_directories(atax PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(atax PRIVATE m polybench_utils)
