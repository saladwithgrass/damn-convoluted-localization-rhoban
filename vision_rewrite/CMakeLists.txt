cmake_minimum_required(VERSION 3.5)
project( DisplayImage )


find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable( run_test src/test.cpp )
target_link_libraries( run_test ${OpenCV_LIBS} )
