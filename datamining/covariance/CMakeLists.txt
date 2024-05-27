add_executable(covariance covariance.cpp)
target_include_directories(covariance PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(covariance PRIVATE m polybench_utils)
