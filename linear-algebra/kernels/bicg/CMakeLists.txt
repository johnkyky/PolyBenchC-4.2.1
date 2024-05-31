add_executable(bicg bicg.cpp)
target_include_directories(bicg PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(bicg PRIVATE m polybench_utils)
