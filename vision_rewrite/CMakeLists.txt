cmake_minimum_required(VERSION 3.5)
project( DisplayImage )


find_package( OpenCV REQUIRED )
find_package( Eigen3 REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Debug)

add_library(prerequisites src/prerequisites/prerequisites.cpp)
add_library(camera_state src/camera_state/camera_state.cpp)
# add_library(white_lines_data src/white_lines_data/white_lines_data.cpp )
add_library(white_lines_detector src/white_lines.cpp)


add_executable( run_test src/test.cpp )
target_link_libraries( run_test prerequisites )
target_link_libraries( run_test white_lines_detector )
# target_link_libraries( run_test camera_state )
target_link_libraries( run_test ${OpenCV_LIBS} )
target_link_libraries( run_test ${Eigen_LIBS} )

