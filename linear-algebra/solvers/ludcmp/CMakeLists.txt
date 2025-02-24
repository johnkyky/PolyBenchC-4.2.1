add_executable(ludcmp ludcmp.cpp)
target_include_directories(ludcmp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ludcmp PRIVATE m polybench_utils)
