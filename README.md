# Cortex: A Modern C++ Deep Learning Framework 

A high-performance deep learning framework crafted with precision in modern C++. Designed to be both educational and practical, Cortex offers a range of tools and functionalities for building and deploying state-of-the-art machine learning models.

Whether you're a seasoned data scientist, a software engineer transitioning into machine learning, or a student eager to explore deep learning's intricacies, Cortex provides a robust foundation for your projects.

Why Cortex? It’s designed to demystify the underlying mathematics of machine learning, featuring clear and straightforward syntax. This approach facilitates a seamless and intuitive transition from mathematical concepts to their practical implementation in software, enhancing both learning and development.
## Features

Cortex stands out with its user-friendly interface and powerful backend, encompassing a wide range of features:

- **Layered Architecture**: Includes fully customizable layers like Dense, Convolutional, and Recurrent layers, each optimized for speed and flexibility.
- **Activation Functions**: Supports various activations including ReLU, Sigmoid, Tanh, and the sophisticated Softmax for handling multi-class classification problems.
- **Loss Functions**: Comprehensive set of loss functions to train models effectively, including MSE, Cross-Entropy, and more.
- **Optimizer Support**: Features advanced optimization algorithms like SGD, Adam, and RMSprop, allowing fine-tuned control over the training process.
- **Backward Propagation**: Efficiently implemented backpropagation to facilitate quick adjustments of model parameters.
- **Extensibility**: Designed with extensibility in mind, allowing researchers and developers to easily add new functionalities or improve existing ones.
- **Highly Documented**: Every function and class is accompanied by clear, concise documentation to help users understand and utilize the library's capabilities fully.

## Getting Started

To get started with Cortex, clone the repository and build the library:

```bash
git clone https://github.com/tamburelloai/cortex.git
cd cortex
mkdir build
cd build
cmake ..
make
```

### Prerequisites

- CMake 3.10 or higher
- A modern C++ compiler supporting C++17
- Optional: BLAS library for optimized linear algebra operations

## Examples

Here’s a quick snippet to get you started with a basic neural network setup using Cortex:

```cpp
#include "Cortex.h"

int main() {
    // Create a model
    Model model;

    // Add layers
    model.addLayer(new DenseLayer(128, 64, new ReLU()));
    model.addLayer(new DenseLayer(64, 10, new Softmax()));

    // Compile the model
    model.compile(new Adam(0.001));

    // Train the model
    model.fit(train_data, train_labels, epochs=10, batch_size=32);

    // Evaluate
    double accuracy = model.evaluate(test_data, test_labels);
    std::cout << "Model accuracy: " << accuracy << "%" << std::endl;

    return 0;
}
```

## Contributing

We welcome contributions from the community, whether it's adding new features, improving documentation, or reporting bugs. Please see our [CONTRIBUTING.md](https://github.com/tamburelloai/cortex/blob/main/CONTRIBUTING.md) for more details on how to contribute.

## License

Cortex is open-source software licensed under the MIT license. See the [LICENSE](https://github.com/tamburelloai/cortex/blob/main/LICENSE) file for more details.

---

**Stay Connected!** Follow our GitHub to stay updated with the latest features and improvements.

---
