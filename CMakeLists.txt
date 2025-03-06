cmake_minimum_required(VERSION 3.5)

project(opencvDNN LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
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
add_executable(ANNHandwriting
    ANNHandwriting.cpp
)
add_executable(ANNHandwritingPredict
    ANNHandwritingPredict.cpp
)
add_executable(ANN_MLP
    ANN_MLP.cpp
)
add_executable(handwritingcnndoubao
    handwritingcnndoubao.cpp
)
add_executable(cvuiHandwriting
    cvuiHandwriting.cpp
)
add_executable(Convolution
    Convolution.cpp
)
add_executable(cvImgturnaround
    cvImgturnaround.cpp
)
add_executable(handwritingPen 
    handwritingPen.cpp
)
add_executable(ANNHandwritingPredictManual 
    ANNHandwritingPredictManual.cpp
)
target_link_libraries(ANNHandwriting PRIVATE  ${OpenCV_LIBS} )
target_link_libraries(ANNHandwritingPredict PRIVATE  ${OpenCV_LIBS} )
target_link_libraries(ANN_MLP PRIVATE  ${OpenCV_LIBS} )
target_link_libraries(handwritingcnndoubao PRIVATE  ${OpenCV_LIBS} )
target_link_libraries(cvuiHandwriting PRIVATE  ${OpenCV_LIBS} )
target_link_libraries(Convolution PRIVATE  ${OpenCV_LIBS} )
target_link_libraries(cvImgturnaround PRIVATE  ${OpenCV_LIBS} )
target_link_libraries(handwritingPen PRIVATE  ${OpenCV_LIBS} )
target_link_libraries(ANNHandwritingPredictManual PRIVATE  ${OpenCV_LIBS} )