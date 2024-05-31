add_executable(mvt mvt.cpp)
target_include_directories(mvt PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(mvt PRIVATE m polybench_utils)
