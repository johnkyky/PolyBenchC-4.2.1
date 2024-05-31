add_executable(3mm 3mm.cpp)
target_include_directories(3mm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(3mm PRIVATE m polybench_utils)
