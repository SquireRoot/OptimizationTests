cmake_minimum_required(VERSION 3.0.0)
project(OptimizationTests VERSION 0.1.0)

# set the c++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# setup some environment variables
set(SRC_DIR ${PROJECT_SOURCE_DIR}/src)

# find all source and header files in the src directory
file(GLOB_RECURSE SOURCE_FILES ${SRC_DIR}/*.cpp)
file(GLOB_RECURSE HEADER_FILES ${SRC_DIR}/*.h)

include(CTest)
enable_testing()

include_directories(${SRC_DIR} ${PROJECT_SOURCE_DIR}/third_party/Eigen/Eigen)
add_executable(${PROJECT_NAME} ${HEADER_FILES} ${SOURCE_FILES})
# target_link_libraries(${PROJECT_NAME} ${LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
