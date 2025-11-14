add_executable(syrk syrk.cpp)
target_include_directories(syrk PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(syrk PRIVATE m polybench_utils)
