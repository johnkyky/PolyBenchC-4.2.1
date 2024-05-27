add_executable(nussinov nussinov.cpp)
target_include_directories(nussinov PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(nussinov PRIVATE m polybench_utils)
