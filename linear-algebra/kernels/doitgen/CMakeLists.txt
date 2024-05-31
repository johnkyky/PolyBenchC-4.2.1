add_executable(doitgen doitgen.cpp)
target_include_directories(doitgen PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(doitgen PRIVATE m polybench_utils)
