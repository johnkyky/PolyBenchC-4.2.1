add_executable(floyd-warshall floyd-warshall.cpp)
target_include_directories(floyd-warshall PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(floyd-warshall PRIVATE m polybench_utils)
