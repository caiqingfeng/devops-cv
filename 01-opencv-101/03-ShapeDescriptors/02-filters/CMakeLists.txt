# cmake needs this line
cmake_minimum_required(VERSION 2.8)

# Define project name
project(opencv_example_project)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)

# If the package has been found, several variables will
# be set, you can find the full list with descriptions
# in the OpenCVConfig.cmake file.
# Print some message showing some of them
message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

if(CMAKE_VERSION VERSION_LESS "2.8.11")
  # Add OpenCV headers location to your include paths
  include_directories(${OpenCV_INCLUDE_DIRS})
endif()

# Declare the executable target built from your sources
add_executable(morphology.out Morphology_1.cpp)
add_executable(hitmiss.out hitmiss.cpp)
add_executable(cnn.out cnn.cpp)
add_executable(loop-filter.out Laplacian.cpp)
add_executable(laplace.out Laplacian.cpp)

# Link your application with OpenCV libraries
target_link_libraries(morphology.out ${OpenCV_LIBS})
target_link_libraries(hitmiss.out ${OpenCV_LIBS})
target_link_libraries(cnn.out ${OpenCV_LIBS})
target_link_libraries(loop-filter.out ${OpenCV_LIBS})
target_link_libraries(laplace.out ${OpenCV_LIBS})
