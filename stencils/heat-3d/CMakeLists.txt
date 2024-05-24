add_executable(heat-3d heat-3d.cpp)
target_include_directories(heat-3d PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(heat-3d PRIVATE m polybench_utils)
