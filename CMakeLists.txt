cmake_minimum_required(VERSION 3.14.0)
project(yolov5_3403_test VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

# set(OpenCV_DIR "/opt/opencv-4.8.1_freetype_video/lib/cmake/opencv4" CACHE PATH "Location of libraries")
SET(OPENCV_DIR "/home/marblech/opencv-4.8.1_freetype_video/lib/cmake/opencv4" CACHE PATH "Location of libraries") 
# set(OPENCV_DIR "/home/marblech/opencv-4.8.1/lib/cmake/opencv4" CACHE PATH "Location of libraries")
set(ACL_SDK_DIR "${PROJECT_SOURCE_DIR}/acl_lib" )
set(ACL_SDK_INCLUDE_DIRS "${ACL_SDK_DIR}/include")
set(ACL_SDK_LIBRARIES_DIRS "${ACL_SDK_DIR}/lib")

find_package(OpenCV REQUIRED PATHS ${OPENCV_DIR} NO_DEFAULT_PATH)

include_directories(${OpenCV_INCLUDE_DIRS} ${ACL_SDK_INCLUDE_DIRS})

link_directories(${OpenCV_LIBRARIES_DIRS} ${ACL_SDK_LIBRARIES_DIRS})

message("OPENCV_INCLUDE_DIRS:${OpenCV_INCLUDE_DIRS}")
message("ACL_SDK_INCLUDE_DIRS:${ACL_SDK_INCLUDE_DIRS}")
message("OPENCV_LIBRARIES:${OpenCV_LIBS}")

add_executable(yolov5_3403_test main.cpp)
add_library(yolov5_3403 SHARED yolov5_3403.hpp yolov5_3403.cpp)

target_link_libraries(yolov5_3403 ${OpenCV_LIBS} ascendcl)
target_link_libraries(yolov5_3403_test yolov5_3403 ${OpenCV_LIBS} ascendcl  -l:libstdc++fs.a)