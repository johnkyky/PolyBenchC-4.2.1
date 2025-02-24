add_executable(gramschmidt gramschmidt.cpp)
target_include_directories(gramschmidt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gramschmidt PRIVATE m polybench_utils)
