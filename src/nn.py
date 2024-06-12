import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initializing weights with random values
        self.weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
        self.weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))

    def feedforward(self, inputs):
        # Calculation of the output of the neural network
        hidden_layer_output = sigmoid(np.dot(inputs, self.weights_input_hidden))
        output = sigmoid(np.dot(hidden_layer_output, self.weights_hidden_output))
        return output

    def train(self, inputs, expected_output, learning_rate):
        # Feedforward
        hidden_layer_output = sigmoid(np.dot(inputs, self.weights_input_hidden))
        output = sigmoid(np.dot(hidden_layer_output, self.weights_hidden_output))

        # Error calculation
        error = expected_output - output

        # Back propagation of the error
        output_delta = error * sigmoid_derivative(output)
        hidden_layer_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_layer_output)

        # Weight update
        self.weights_hidden_output += learning_rate * np.outer(hidden_layer_output, output_delta)
        self.weights_input_hidden += learning_rate * np.outer(inputs, hidden_layer_delta)

