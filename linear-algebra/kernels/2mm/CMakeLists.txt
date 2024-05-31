add_executable(2mm 2mm.cpp)
target_include_directories(2mm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(2mm PRIVATE m polybench_utils)
