cmake_minimum_required(VERSION 3.5)
project(Q-Learning)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set( CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}" ${CMAKE_MODULE_PATH} )
set( OpenCV_DIR ${CMAKE_SOURCE_DIR}/opencv )
set(SOURCE_FILES main.cpp utils/utils.cpp utils/utils.h)

FIND_PACKAGE( OpenCV REQUIRED )

INCLUDE_DIRECTORIES( ${OpenCV_INCLUDE_DIRS}  )
LINK_DIRECTORIES( ${CMAKE_SOURCE_DIR}/lib/curl ${CMAKE_SOURCE_DIR}/lib/robot ${PCL_LIBRARY_DIRS} ${KinectSDK2_LIBRARY_DIRS} )

add_executable(Q-Learning ${SOURCE_FILES})

TARGET_LINK_LIBRARIES(${PROJECT_NAME} ${OpenCV_LIBS} legacy_stdio_definitions.lib )


# Set optimized building:
IF(CMAKE_COMPILER_IS_GNUCXX AND NOT CMAKE_BUILD_TYPE MATCHES "Debug")
	SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
ENDIF()
