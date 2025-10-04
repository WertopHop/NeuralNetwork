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


## Dataset

The project uses the **MNIST** dataset, which contains:
- 70,000 images of handwritten digits (0-9)
- Image size: 28x28 pixels
- Training set: 60,000 images
- Data normalization: pixel values ​​are divided by 255 to convert them to the range [0, 1]

**Python version**: Loads the dataset via OpenML (dataset ID: 554)

**C++ version**: Reads the dataset from an ARFF file (`mnist_784.arff`)

---

## Model Accuracy

### Python version

- **Accuracy on the testing set Sample Loss**: 97.3%

### C++ Version

- **Accuracy on the testing set Sample Loss**: 99.2%
