# Set the minimum cmake version
cmake_minimum_required(VERSION 3.10.0)

# set the project name and version
project(OptimizationTests VERSION 0.1.0)

# set the c++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# setup some environment variables
set(SRC_DIR ${PROJECT_SOURCE_DIR}/src)
set(TEST_DIR ${PROJECT_SOURCE_DIR}/test)

# find all source and header files in the src directory
file(GLOB_RECURSE SOURCE_FILES ${SRC_DIR}/*.cpp)
file(GLOB_RECURSE HEADER_FILES ${SRC_DIR}/*.h)

# find all unit test source files
file(GLOB_RECURSE TEST_SOURCE_FILES ${SRC_DIR}/*.cpp)
file(GLOB_RECURSE TEST_HEADER_FILES ${SRC_DIR}/*.h)

# setup the cmake module path
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake")

# Add other platform dependent stuff, mostly libraries
if (WIN32) # if we are on windows
    message("Windows detected")

elseif(UNIX AND NOT APPLE) # if we are on unix
    message("Unix detected")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUF} -Wall")

else()

endif(WIN32)

# release
include_directories(${SRC_DIR} ${PROJECT_SOURCE_DIR}/third_party/Eigen/Eigen)
add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} ${LIBS})