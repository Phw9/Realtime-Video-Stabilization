cmake_minimum_required(VERSION 2.8.12)

PROJECT(VideoStabilization)

######################## EDIT IF REQUIRED ####################
# If CMake is not able to find the path to OpenCV installation then 
# uncomment the line below and specify the path to OpenCV directory 
# (i.e. the path to the OpenCVConfig.cmake file). Check the examples given below.
#SET(OpenCV_DIR Enter-the-path-of-OpenCV-installation-on-your-system)


################### OpenCV_DIR Examples  #####################

### MACOS : /usr/local/Cellar/opencv/3.4.3/share/OpenCV/

### UBUNTU : /usr/local/share/OpenCV/

### WINDOWS : C:\Users\yourname\Documents\opencv-3.4.3\build\install

##############################################################

#############  Common Instructions for all Users  ############
find_package( OpenCV REQUIRED )

include_directories( 
  ${OpenCV_INCLUDE_DIRS}
  ${PROJECT_SOURCE_DIR}/include
)

add_library(mine SHARED
  ${PROJECT_SOURCE_DIR}/src/videostab.cpp
)

target_link_libraries(mine
  ${EIGEN3_LIBS}
  ${GLUT_LIBRARY}
  ${OpenCV_LIBS}
  ${OPENGL_LIBRARIES}
  ${Pangolin_LIBRARIES}
)

add_executable(VideoStabilization
  ${PROJECT_SOURCE_DIR}/main.cpp
)

target_link_libraries(VideoStabilization
  ${OpenCV_LIBS}
  ${EIGEN3_LIBS}
  ${GLUT_LIBRARY}
  ${OpenCV_LIBS}
  ${OPENGL_LIBRARIES}
  ${Pangolin_LIBRARIES}
  mine
)

target_COMPILE_OPTIONS(VideoStabilization PUBLIC -g -Wall)

