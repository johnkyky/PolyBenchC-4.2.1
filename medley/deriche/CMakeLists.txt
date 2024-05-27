add_executable(deriche deriche.cpp)
target_include_directories(deriche PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(deriche PRIVATE m polybench_utils)
