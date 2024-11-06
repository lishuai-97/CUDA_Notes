cmake_minimum_required (VERSION 3.26)
 
project (cmake_main)
 
enable_language(CXX)
enable_language(CUDA)
 
# set(CMAKE_CUDA_COMPILER_WORKS TRUE)
# set(CMAKE_CXX_COMPILER_WORKS TRUE)
 
find_package(CUDAToolkit REQUIRED)
 
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
 
#set(CMAKE_CUDA_ARCHITECTURES 75)
 
set(INCLUDE_MSVC "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.29.30133")
set(INCLUDE_CMAKE "C:/Program Files (x86)/Windows Kits/10/Include/10.0.19041.0/ucrt")
 
set(INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include" "${INCLUDE_MSVC}/include" "${INCLUDE_CMAKE}")
set(SRC_DIR "${CMAKE_SOURCE_DIR}/src")
 
file(GLOB SRC_FILES "*.cpp" "${SRC_DIR}/*.c" "${SRC_DIR}/*.cpp" "${SRC_DIR}/*.cc" "${SRC_DIR}/*.cxx" "${SRC_DIR}/*.cu")
 
message(STATUS "Src files: ${SRC_FILES}")
 
if (CMAKE_CUDA_COMPILER)
    message(STATUS "nvcc path : ${CMAKE_CUDA_COMPILER}")
else ()
    message(WARNING "nvcc not found. Please check CUDA is installed correctly!")
endif ()
 
set(LINK_LIBRARIES_KERNEL32 "C:/Program Files (x86)/Windows Kits/10/Lib/10.0.19041.0/um/x64")
set(LINK_LIBRARIES_MSVC     "C:/Program Files (x86)/Microsoft Visual Studio/2019/Community/VC/Tools/MSVC/14.29.30133/lib/x64")
set(LINK_LIBRARIES_UCRT     "C:/Program Files (x86)/Windows Kits/10/Lib/10.0.19041.0/ucrt/x64")
 
# link directories
link_directories("${LINK_LIBRARIES_KERNEL32}" "${LINK_LIBRARIES_MSVC}" "${LINK_LIBRARIES_UCRT}") 
include_directories("${LINK_LIBRARIES_KERNEL32}" "${LINK_LIBRARIES_MSVC}" "${LINK_LIBRARIES_UCRT}") 
 
add_executable(${PROJECT_NAME})
 
target_sources(${PROJECT_NAME} PRIVATE ${SRC_FILES})
target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIR})
