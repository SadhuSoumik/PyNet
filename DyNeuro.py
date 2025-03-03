import numpy as np
import random

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

class Layer:
    def __init__(self, input_size, output_size):
        # Xavier/Glorot initialization
        scale = np.sqrt(6.0 / (input_size + output_size))
        self.weights = np.random.uniform(-scale, scale, (input_size, output_size))
        self.biases = np.zeros((output_size, 1))  # Shape: (output_size, 1)
        self.pre_activation = None
        self.activation = None

    def forward(self, input):
        # Compute pre-activation: weights.T dot input + biases
        self.pre_activation = np.dot(self.weights.T, input) + self.biases
        self.activation = sigmoid(self.pre_activation)
        return self.activation

class Adam:
    def __init__(self, layer_sizes, learning_rate):
        self.learning_rate = learning_rate
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.t = 0
        self.m_weights = []
        self.v_weights = []
        self.m_biases = []
        self.v_biases = []
        for i in range(len(layer_sizes) - 1):
            self.m_weights.append(np.zeros((layer_sizes[i], layer_sizes[i+1])))
            self.v_weights.append(np.zeros((layer_sizes[i], layer_sizes[i+1])))
            self.m_biases.append(np.zeros((layer_sizes[i+1], 1)))
            self.v_biases.append(np.zeros((layer_sizes[i+1], 1)))

    def update(self, layers, weight_gradients, bias_gradients):
        self.t += 1
        # Bias-corrected learning rate
        lr_t = self.learning_rate * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)
        for i in range(len(layers)):
            # Update weights
            self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * weight_gradients[i]
            self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * (weight_gradients[i]**2)
            m_hat = self.m_weights[i] / (1 - self.beta1**self.t)
            v_hat = self.v_weights[i] / (1 - self.beta2**self.t)
            layers[i].weights -= lr_t * m_hat / (np.sqrt(v_hat) + self.epsilon)
            # Update biases
            self.m_biases[i] = self.beta1 * self.m_biases[i] + (1 - self.beta1) * bias_gradients[i]
            self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * (bias_gradients[i]**2)
            m_hat = self.m_biases[i] / (1 - self.beta1**self.t)
            v_hat = self.v_biases[i] / (1 - self.beta2**self.t)
            layers[i].biases -= lr_t * m_hat / (np.sqrt(v_hat) + self.epsilon)

class NeuralNetwork:
    def __init__(self, sizes, max_hidden_neurons):
        """
        sizes: list of layer sizes, e.g. [2, 4, 1]
        max_hidden_neurons: maximum allowed neurons in the hidden layer(s)
        """
        assert len(sizes) >= 3, "Network must have at least input, one hidden, and output layers"
        self.architecture = sizes.copy()
        self.max_hidden_neurons = max_hidden_neurons
        self.layers = []
        for i in range(len(sizes) - 1):
            self.layers.append(Layer(sizes[i], sizes[i+1]))

    def forward(self, input):
        current = input
        for layer in self.layers:
            current = layer.forward(current)
        return current

    def add_neuron(self):
        # Add a neuron to the first hidden layer (index 0)
        hidden_layer_idx = 0
        hidden_neurons = self.layers[hidden_layer_idx].biases.shape[0]
        if hidden_neurons >= self.max_hidden_neurons:
            print(f"Max hidden neurons ({self.max_hidden_neurons}) reached, skipping growth.")
            return

        # Update architecture: increase neuron count in hidden layer by one
        self.architecture[hidden_layer_idx + 1] += 1
        input_size = self.layers[hidden_layer_idx].weights.shape[0]
        # Output layer (following the hidden layer) weight shape: (hidden_neurons, output_size)
        output_size = self.layers[hidden_layer_idx+1].weights.shape[1]

        # --- Update first hidden layer ---
        # New weight matrix with shape (input_size, hidden_neurons+1)
        new_hidden_weights = np.zeros((input_size, hidden_neurons + 1))
        new_hidden_weights[:, :hidden_neurons] = self.layers[hidden_layer_idx].weights
        scale = np.sqrt(6.0 / (input_size + hidden_neurons + 1))
        new_hidden_weights[:, hidden_neurons] = np.random.uniform(-scale, scale, input_size)
        self.layers[hidden_layer_idx].weights = new_hidden_weights

        # Update biases: new shape (hidden_neurons+1, 1)
        new_biases = np.zeros((hidden_neurons + 1, 1))
        new_biases[:hidden_neurons, :] = self.layers[hidden_layer_idx].biases
        self.layers[hidden_layer_idx].biases = new_biases

        # --- Update output layer ---
        # New weight matrix with shape (hidden_neurons+1, output_size)
        new_output_weights = np.zeros((hidden_neurons + 1, output_size))
        new_output_weights[:hidden_neurons, :] = self.layers[hidden_layer_idx+1].weights
        scale = np.sqrt(6.0 / (hidden_neurons + 1 + output_size))
        new_output_weights[hidden_neurons, :] = np.random.uniform(-scale, scale, output_size)
        self.layers[hidden_layer_idx+1].weights = new_output_weights

        print(f"Added neuron to hidden layer. New size: {hidden_neurons + 1}")

    def backpropagate(self, input, target):
        # Forward pass: store all activations for each layer
        activations = [input]
        current = input
        for layer in self.layers:
            current = layer.forward(current)
            activations.append(current)
        output = activations[-1]
        error = output - target

        weight_gradients = []
        bias_gradients = []
        # Derivative of MSE loss: 2 * (output - target)
        delta = 2 * error

        # Backpropagation loop (from output layer to first hidden layer)
        for i in reversed(range(len(self.layers))):
            pre_activation = self.layers[i].pre_activation
            sigmoid_deriv = sigmoid_derivative(pre_activation)
            delta_with_sigmoid = delta * sigmoid_deriv
            # Weight gradient: outer product of activation from previous layer and delta (transposed)
            weight_gradient = np.dot(activations[i], delta_with_sigmoid.T)
            weight_gradients.insert(0, weight_gradient)
            bias_gradients.insert(0, delta_with_sigmoid)
            if i > 0:
                delta = np.dot(self.layers[i].weights, delta_with_sigmoid)
        return weight_gradients, bias_gradients

    def train(self, inputs, targets, epochs, batch_size):
        assert len(inputs) == len(targets), "Number of inputs and targets must match"
        optimizer = Adam(self.architecture, 0.01)
        prev_loss = float('inf')
        patience = 30  # epochs to wait before growing
        patience_counter = 0
        loss_threshold = 0.0001
        num_samples = len(inputs)
        num_batches = (num_samples + batch_size - 1) // batch_size

        for epoch in range(epochs):
            indices = list(range(num_samples))
            random.shuffle(indices)
            total_loss = 0.0

            for batch in range(num_batches):
                start_idx = batch * batch_size
                end_idx = min(start_idx + batch_size, num_samples)
                current_batch_size = end_idx - start_idx

                # Initialize batch gradients
                batch_weight_gradients = [np.zeros_like(layer.weights) for layer in self.layers]
                batch_bias_gradients = [np.zeros_like(layer.biases) for layer in self.layers]

                for idx in indices[start_idx:end_idx]:
                    weight_gradients, bias_gradients = self.backpropagate(inputs[idx], targets[idx])
                    for j in range(len(self.layers)):
                        batch_weight_gradients[j] += weight_gradients[j] / current_batch_size
                        batch_bias_gradients[j] += bias_gradients[j] / current_batch_size
                    output = self.forward(inputs[idx])
                    total_loss += np.mean((output - targets[idx])**2)

                optimizer.update(self.layers, batch_weight_gradients, batch_bias_gradients)

            avg_loss = total_loss / num_samples

            if (prev_loss - avg_loss) < loss_threshold:
                patience_counter += 1
                if patience_counter >= patience:
                    self.add_neuron()
                    patience_counter = 0
                    optimizer = Adam(self.architecture, 0.01)  # Reinitialize optimizer
            else:
                patience_counter = 0

            prev_loss = avg_loss
            if epoch % 100 == 0 or epoch == epochs - 1:
                print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")

if __name__ == "__main__":
    # XOR dataset: each input is a (2,1) vector, target is (1,1)
    inputs = [
        np.array([[0.0], [0.0]]),
        np.array([[0.0], [1.0]]),
        np.array([[1.0], [0.0]]),
        np.array([[1.0], [1.0]])
    ]
    targets = [
        np.array([[0.0]]),
        np.array([[1.0]]),
        np.array([[1.0]]),
        np.array([[0.0]])
    ]

    # Initialize network: 2 inputs, 4 hidden neurons, 1 output, with a max of 10 hidden neurons.
    nn = NeuralNetwork([2, 4, 1], max_hidden_neurons=10)
    nn.train(inputs, targets, epochs=1000, batch_size=2)

    print("\nTesting the trained network:")
    for inp in inputs:
        output = nn.forward(inp)
        print(f"Input: {inp.ravel()}, Output: {output.ravel()[0]:.4f}")
