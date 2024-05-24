add_executable(fdtd-2d fdtd-2d.cpp)
target_include_directories(fdtd-2d PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(fdtd-2d PRIVATE m polybench_utils)
