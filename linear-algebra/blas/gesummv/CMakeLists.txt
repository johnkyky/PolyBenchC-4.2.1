add_executable(gesummv gesummv.cpp)
target_include_directories(gesummv PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(gesummv PRIVATE m polybench_utils)
