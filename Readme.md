# DyNeuro.py - Dynamic Neural Network Implementation

This repository contains a Python implementation of a dynamic neural network that can grow by adding neurons to its hidden layer during training. The network uses the Adam optimizer and is trained on the XOR dataset.

## Features

- **Dynamic Growth**: The network can add neurons to its hidden layer if the loss does not improve significantly over a number of epochs.
- **Adam Optimizer**: Utilizes the Adam optimization algorithm for efficient training.
- **XOR Dataset**: The network is trained on the XOR problem, a classic example in neural network training.

## Code Analysis

### Key Components

1. **Layer Class**: Represents a layer in the neural network with weights and biases initialized using Xavier/Glorot initialization.
2. **Adam Class**: Implements the Adam optimization algorithm for updating weights and biases.
3. **NeuralNetwork Class**: Manages the network architecture, forward propagation, backpropagation, and dynamic neuron addition.
4. **Training**: The network is trained using mini-batch gradient descent with the Adam optimizer. If the loss does not improve significantly, a neuron is added to the hidden layer.

### Dynamic Neuron Addition

The network can add neurons to its hidden layer if the loss does not improve by a specified threshold over a number of epochs (`patience`). This allows the network to adapt its capacity based on the complexity of the task.

### Training Process

- **Initialization**: The network is initialized with a specified architecture and a maximum number of hidden neurons.
- **Forward Pass**: Computes the activations for each layer.
- **Backpropagation**: Computes gradients for weights and biases.
- **Optimization**: Updates weights and biases using the Adam optimizer.
- **Dynamic Growth**: Adds neurons to the hidden layer if the loss does not improve significantly.

## Cloning and Running the Code

To clone and run the code, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/SadhuSoumik/PyNet.git
   cd DyNeuro
   ```
   ### Install Dependencies:
```bash
pip install numpy
```
### Run the Script:
```bash
python DyNeuro.py
```
### Observe the Output:
The script will print the loss at every 100 epochs and the final outputs for the XOR inputs after training.

## Example Output

```bash
Epoch 0: Loss = 0.250000
Epoch 100: Loss = 0.062500
...
Epoch 900: Loss = 0.000123
Epoch 999: Loss = 0.000120

Testing the trained network:
Input: [0. 0.], Output: 0.0012
Input: [0. 1.], Output: 0.9988
Input: [1. 0.], Output: 0.9988
Input: [1. 1.], Output: 0.0012
```


   
