# Neural Networks

**This educational repository neural network development, starting from basic parameter tuning and progressing to optimized implementations that enhance model performance and accuracy. By building a neural network from scratch in Python without high-level frameworks, you’ll gain a deep understanding of core machine learning concepts.**

## Python Neural Network from Scratch

This is a neural network implementation from scratch in Python, without using high-level machine learning frameworks (such as TensorFlow or PyTorch). The project demonstrates a complete understanding of neural networks, including feedforward and backpropagation.

### Technologies and Tools

- **Programming Language**: Python
- **Libraries**:
- `numpy` - for performing mathematical operations with arrays and matrices
- `openml` - for loading the MNIST dataset
- `pandas` - for working with data in tabular format

### Neural Network Architecture

The neural network has a customizable architecture. By default, it uses:
- **Input Layer**: 784 neurons (for 28x28 pixel images)
- **Hidden Layers**:
- First Hidden Layer: 256 neurons
- Second Hidden Layer: 32 neurons
- **Output Layer**: 10 neurons (for classifying digits 0-9)

**Activation Function**: Sigmoid
**Loss Function**: Mean Squared Error (MSE)

### Main components:

1. **Weight initialization**: Random initialization of weights with a normal distribution
2. **Feedforward**: Compute the network output values
3. **Train**: Implement the backpropagation algorithm with gradient descent
4. **Save model**: Save the trained weights and biases to the file `neural_network_weights.npz`
5. **Load model**: Automatically load the saved weights when making a prediction

### Installing dependencies

```bash
pip install numpy openml pandas
```

### Running

```bash
python python_neuro/neuro.py
```

---

## C++ version

### Technologies and tools

- **Programming language**: C++11/14
- **Libraries**:
- `<vector>` - for working with dynamic arrays
- `<cmath>` - for mathematical functions (exp)
- `<fstream>` - for working with files (saving/loading a model and reading a dataset)
- `<omp.h>` - OpenMP for multithreading and parallel computing

### Neural network architecture

The C++ version has a more compact architecture to improve training speed:
- **Input layer**: 784 neurons (for 28x28 pixel images)
- **Hidden layers**:
- First hidden layer: 16 neurons
- Second hidden layer: 16 neurons
- **Output layer**: 10 neurons (for digit classification) 0-9)

**Activation Function**: Sigmoid
**Loss Function**: Mean Squared Error (MSE)

### Main Components:

1. **Weight Initialization**: Random initialization of weights in the range [-1.0, 1.0]
2. **Feedforward**: Computing network outputs using parallel computing
3. **Train**:
- Mini-batch gradient descent implementation
- Parallel processing of examples in a batch using OpenMP
- Accumulation of gradients and weight updates after each batch
4. **Save Model**: Saving trained weights and biases to the binary file `neural_model.bin`
5. **Load Model**: Automatically loading saved weights during network initialization

### Implementation Features

- ✅ **Multithreading**: Using OpenMP to parallelize feedforward and training computations
- ✅ **Mini-batch training**: Processing data in batches for more stable convergence
- ✅ **Binary storage**: Efficiently saving the model in binary format
- ✅ **Performance**: Significantly faster than the Python version thanks to compilation and parallelization
- ✅ **Working with ARFF**: Reading the MNIST dataset from ARFF format

### Compilation

```bash
# With OpenMP support
g++ -std=c++11 -fopenmp -O3 -o neural_network C++_neuro/Neuro.cpp

# Or using clang
clang++ -std=c++11 -fopenmp -O3 -o neural_network C++_neuro/Neuro.cpp
```

### Run

```bash
# Make sure the mnist_784.arff file is in the same directory
./neural_network

# The program will ask if you want to train the model
# Enter 'y' to train or just press Enter to load an existing model
```

---

## PyTorch Neural Network

This is a modern implementation of a neural network using the **PyTorch** framework. This version demonstrates best practices in deep learning, including automatic differentiation, GPU acceleration, and advanced training techniques.

### Technologies and Tools

- **Programming Language**: Python 3
- **Framework**: PyTorch
- **Libraries**:
  - `torch` - PyTorch deep learning framework
  - `torch.nn` - neural network modules
  - `torch.optim` - optimization algorithms
  - `openml` - for loading datasets
  - `pandas` - for data manipulation
  - `numpy` - for numerical operations
  - `scikit-learn` - for data preprocessing and train-test split
  - `torch.utils.data` - for data loading utilities

### Neural Network Architecture

The PyTorch implementation uses a flexible, modular architecture:
- **Input Layer**: 784 neurons (for 28x28 pixel images)
- **Hidden Layers**:
  - First Hidden Layer: 256 neurons + ReLU + Dropout (0.2)
  - Second Hidden Layer: 16 neurons + ReLU + Dropout (0.2)
- **Output Layer**: 10 neurons (for classifying digits 0-9)

**Activation Function**: ReLU (Rectified Linear Unit)  
**Loss Function**: CrossEntropyLoss (for multi-class classification)  
**Optimizer**: Adam (Adaptive Moment Estimation)  
**Regularization**: Dropout (20% dropout rate)

### Advantages over From-Scratch Implementation

- ✅ **Automatic Differentiation**: No manual backpropagation implementation needed
- ✅ **GPU Acceleration**: Significant speedup on CUDA-enabled devices
- ✅ **Advanced Optimizers**: Adam optimizer with adaptive learning rates
- ✅ **Better Activation Functions**: ReLU instead of Sigmoid (reduces vanishing gradients)
- ✅ **Regularization**: Dropout prevents overfitting
- ✅ **Data Preprocessing**: Comprehensive StandardScaler normalization
- ✅ **Batch Processing**: Efficient mini-batch training with DataLoader
- ✅ **Production Ready**: Industry-standard framework with extensive support
- 
### Installing Dependencies

```bash
pip install torch torchvision openml pandas numpy scikit-learn
```

### Running

```bash
python PyTorch_Neural_Network/Neuro_PyTorch.py
```

## Dataset

The project uses the **MNIST** dataset (OpenML ID: 554), which contains:
- 70,000 images of handwritten digits (0-9)
- Image size: 28x28 pixels (784 features when flattened)
- Training set: 60,000 images
- Test set: 10,000 images
- Normalized pixel values in range [0, 1]

**Python from-scratch version**: Loads via OpenML, uses 60,000 samples for training  
**C++ version**: Reads from ARFF file  
**PyTorch version**: Loads via OpenML with 85%/15% train-test split, applies StandardScaler normalization


---


## Model Accuracy

### Python version

- **Accuracy on the testing set Sample Loss**: 97.3%

### C++ Version

- **Accuracy on the testing set Sample Loss**: 99.2%

### Pytorch Version

- **Accuracy on the testing set Sample Loss**: 96.3%

  
---


## Author

**WertopHop** - [GitHub](https://github.com/WertopHop)
