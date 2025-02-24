add_executable(durbin durbin.cpp)
target_include_directories(durbin PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(durbin PRIVATE m polybench_utils)
