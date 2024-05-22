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
        src/activations/relu.cpp
        src/activations/softmax.cpp
        src/matrix.cpp
        src/parameter.cpp
        src/one_hot_encoder.cpp
        src/dataset.cpp
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
        examples/MLP.cpp
        include/data_utils/cortex_dataset.h
        src/data_utils/cortex_dataset.cpp
        src/dataloader.cpp
        include/data_utils/dataloader.h
        include/layers/layer.h
        include/model.h
        include/activations/relu.h
        src/activations/relu.cpp
        include/optimizers/optimizer.h
        include/optimizers/sgd.h
        include/loss_functions/loss.h
        include/loss_functions/mse_loss.h
        include/loss_functions/crossentropy_loss.h
        include/activations/softmax.h
        src/activations/softmax.cpp
        tests/ActivationsTest.cpp
        include/optimizers/adam.h
)
target_link_libraries(AllTests cortex_layers gtest gtest_main pthread)

# Define a test to be run by ctest
add_test(NAME AllTesting COMMAND AllTests)
