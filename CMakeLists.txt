cmake_minimum_required(VERSION 3.5)

project(HandwritingDemo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /source-charset:utf-8 /execution-charset:gbk")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /source-charset:utf-8 /execution-charset:gbk")
endif()

include_directories(${CMAKE_CURRENT_SOURCE_DIR})
	
find_package(OpenCV 4 REQUIRED )
include_directories(
    ${OpenCV_INCLUDE_DIRS}/include
    )
message(STATUS "-OpenCV_INCLUDE_DIRS--- = ${OpenCV_INCLUDE_DIRS}/include")
# -------------- Compile CUDA for FP16 inference if needed  ------------------#
option(USE_CUDA "Enable CUDA support" ON)
if (NOT APPLE AND USE_CUDA)
    find_package(CUDA REQUIRED)
    include_directories(${CUDA_INCLUDE_DIRS})
    add_definitions(-DUSE_CUDA)
else ()
    set(USE_CUDA OFF)
endif ()
add_executable(cvuiHandwritingPredication
    cvuiHandwritingPredication.cpp
)

add_executable(cvuiHandwritingTrainDemo 
    cvuiHandwritingTrainDemo.cpp
)
add_executable(ANNHandwritingTrain
    ANNHandwritingTrain.cpp
)
target_link_libraries(cvuiHandwritingPredication PRIVATE  ${OpenCV_LIBS} )
target_link_libraries(cvuiHandwritingTrainDemo PRIVATE  ${OpenCV_LIBS} )
target_link_libraries(ANNHandwritingTrain PRIVATE  ${OpenCV_LIBS} )