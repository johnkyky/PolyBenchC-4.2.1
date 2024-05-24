add_executable(adi adi.cpp)
target_include_directories(adi PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(adi PRIVATE m polybench_utils)
