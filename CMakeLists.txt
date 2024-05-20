cmake_minimum_required(VERSION 3.28)
project(cortex)

set(CMAKE_CXX_STANDARD 17)

# Correct include directory
include_directories(include)

# Enable testing
enable_testing()

# Google Test
add_subdirectory(GTest)

# Define the library for layer functionalities
add_library(
        cortex_layers
        src/layers/linear.cpp
        src/matrix.cpp
        src/parameter.cpp
)
target_include_directories(cortex_layers PUBLIC include)

# Main executable
add_executable(
        cortex
        src/main.cpp
)
target_link_libraries(cortex cortex_layers)

# Test executable
add_executable(
        AllTests
        tests/test_main.cpp
        tests/MatrixTest.cpp
        tests/ParameterTest.cpp
        tests/LinearTest.cpp
        src/optimizers/sgd.cpp
        include/optimizers/sgd.h
        src/optimizers/optimizer.cpp
        include/optimizers/optimizer.h
        src/optimizers/adam.cpp
        include/optimizers/adam.h
)
target_link_libraries(AllTests cortex_layers gtest gtest_main pthread)

# Define a test to be run by ctest
add_test(NAME AllTesting COMMAND AllTests)
