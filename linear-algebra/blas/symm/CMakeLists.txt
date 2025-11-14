add_executable(symm symm.cpp)
target_include_directories(symm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(symm PRIVATE m polybench_utils)
