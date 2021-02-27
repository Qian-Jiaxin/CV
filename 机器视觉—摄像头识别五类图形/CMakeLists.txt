cmake_minimum_required(VERSION 2.8)

project(detect_match)

set(CMAKE_CXX_STANDARD 11)

find_PACKAGE(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(detect_match detect_match.cc)

target_link_libraries(detect_match ${OpenCV_LIBS})
