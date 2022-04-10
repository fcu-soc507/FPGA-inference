cmake_minimum_required(VERSION 3.5.1)
project(sort-cpp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)

# Try to find OpenCV
#find_package(OpenCV REQUIRED core highgui imgproc)
#if (OpenCV_FOUND)
#    # If the package has been found, several variables will
#    # be set, you can find the full list with descriptions
#    # in the OpenCVConfig.cmake file.
#    # Print some message showing some of them
#    message(STATUS "OpenCV library status:")
#    message(STATUS "    version: ${OpenCV_VERSION}")
#    message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
#else ()
#    message(FATAL_ERROR "Could not locate OpenCV")
#endif()

# Try to find Boost
find_package(Boost COMPONENTS program_options filesystem REQUIRED)
if(Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIRS})
else ()
    message(FATAL_ERROR "Could not locate Boost")
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(/usr/include/opencv4/)

file(GLOB SOURCE_FILES src/*.cpp)

add_library(${CMAKE_PROJECT_NAME} SHARED ${SOURCE_FILES})
add_executable(${CMAKE_PROJECT_NAME}-bin ${SOURCE_FILES} main.cpp)

target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/include)

target_link_libraries (
        ${CMAKE_PROJECT_NAME}
        ${Boost_LIBRARIES}
	opencv_core
	opencv_highgui
	opencv_imgproc
	opencv_imgcodecs
)

target_link_libraries (
        ${CMAKE_PROJECT_NAME}-bin
        ${Boost_LIBRARIES}
	opencv_core
	opencv_highgui
	opencv_imgproc
	opencv_imgcodecs
)

