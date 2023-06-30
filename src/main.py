import RPi.GPIO as GPIO
import numpy as np
import json

from nn import NeuralNetwork

with open("training_data.json", "r") as file:
    data = json.load(file)

if __name__ == "__main__":
    led_pins = [12, 23, 32, 40]
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(led_pins, GPIO.OUT)
    
    training_data = data["training_xor"] 
    neural_network = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
    
    try:
        # Training model
        for _ in range(10000):
            for data in training_data:
                inputs = np.array(data["input"])
                expected_output = np.array(data["output"])
                neural_network.train(inputs, expected_output, learning_rate=0.1)

        outputs = []
        for data in training_data:
            inputs = np.array(data["input"])
            output = neural_network.feedforward(inputs)[0]
            print(f"{inputs} -> {output}")
            outputs.append(round(output))
        
        while True:
            # print(outputs)
            for output, pin in zip(outputs, led_pins):
                GPIO.output(pin, GPIO.HIGH if output else GPIO.LOW)
 
    except KeyboardInterrupt:
        pass
    finally:
        GPIO.output(led_pins, GPIO.LOW)
        GPIO.cleanup()

