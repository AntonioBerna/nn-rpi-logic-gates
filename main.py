import RPi.GPIO as GPIO
import numpy as np

led_pins = [12, 23, 32, 33]

training_and = [
    {'input': [0, 0], 'output': 0},
    {'input': [0, 1], 'output': 0},
    {'input': [1, 0], 'output': 0},
    {'input': [1, 1], 'output': 1},
]

training_or = [
    {'input': [0, 0], 'output': 0},
    {'input': [0, 1], 'output': 1},
    {'input': [1, 0], 'output': 1},
    {'input': [1, 1], 'output': 1},
]

training_nand = [
    {'input': [0, 0], 'output': 1},
    {'input': [0, 1], 'output': 1},
    {'input': [1, 0], 'output': 1},
    {'input': [1, 1], 'output': 0},
]

training_nor = [
    {'input': [0, 0], 'output': 1},
    {'input': [0, 1], 'output': 0},
    {'input': [1, 0], 'output': 0},
    {'input': [1, 1], 'output': 0},
]

training_xor = [
    {'input': [0, 0], 'output': 0},
    {'input': [0, 1], 'output': 1},
    {'input': [1, 0], 'output': 1},
    {'input': [1, 1], 'output': 0},
]

training_data = training_nand

# Inizializzazione dei pin GPIO
GPIO.setmode(GPIO.BOARD)
GPIO.setup(led_pins, GPIO.OUT)

# Funzione di attivazione sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Funzione di derivata della sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Classe per la rete neurale
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Inizializzazione dei pesi con valori casuali
        self.weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
        self.weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))

    def feedforward(self, inputs):
        # Calcolo dell'output della rete neurale
        hidden_layer_output = sigmoid(np.dot(inputs, self.weights_input_hidden))
        output = sigmoid(np.dot(hidden_layer_output, self.weights_hidden_output))
        return output

    def train(self, inputs, expected_output, learning_rate):
        # Feedforward
        hidden_layer_output = sigmoid(np.dot(inputs, self.weights_input_hidden))
        output = sigmoid(np.dot(hidden_layer_output, self.weights_hidden_output))

        # Calcolo dell'errore
        error = expected_output - output

        # Retropropagazione dell'errore
        output_delta = error * sigmoid_derivative(output)
        hidden_layer_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_layer_output)

        # Aggiornamento dei pesi
        self.weights_hidden_output += learning_rate * np.outer(hidden_layer_output, output_delta)
        self.weights_input_hidden += learning_rate * np.outer(inputs, hidden_layer_delta)

if __name__ == "__main__":
    input_size = 2
    hidden_size = 4
    output_size = 1
    neural_network = NeuralNetwork(input_size, hidden_size, output_size)

    try:
        # Addestramento del modello
        for _ in range(10000):
            for data in training_data:
                inputs = np.array(data['input'])
                expected_output = np.array(data['output'])
                neural_network.train(inputs, expected_output, learning_rate=0.1)
        
        outputs = []
        for data in training_data:
            inputs = np.array(data['input'])
            output = neural_network.feedforward(inputs)[0]
            print(f"Input: {inputs}, Output: {output}")
            outputs.append(round(output))
        
        while True:
            # print(outputs)
            for i in range(len(outputs)):
                if outputs[i]:
                    GPIO.output(led_pins[i], GPIO.HIGH)
                else:
                    GPIO.output(led_pins[i], GPIO.LOW)
    except KeyboardInterrupt:
        pass
    finally:
        GPIO.output(led_pins, GPIO.LOW)
        GPIO.cleanup()

