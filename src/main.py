import RPi.GPIO as GPIO
import json
import time

from nn import NeuralNetwork, np

if __name__ == "__main__":
    led_pins = [12, 23, 32, 40]
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(led_pins, GPIO.OUT)

    try:
        with open("assets/training_data.json", "r") as file:
            training_data = json.load(file)
            for key in training_data:
                neural_network = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
                model = training_data[key]
                print(f"{key[9:]} model in progress.")
                for _ in range(10000):
                    for features in model:
                        inputs = np.array(features["input"])
                        expected_output = np.array(features["output"])
                        neural_network.train(inputs, expected_output, learning_rate=0.1)
        
                outputs = []
                for features in model:
                    inputs = np.array(features["input"])
                    output = neural_network.feedforward(inputs)[0]
                    print(f"{inputs} -> {output}")
                    outputs.append(round(output))
    
                for output, pin in zip(outputs, led_pins):
                    GPIO.output(pin, GPIO.HIGH if output else GPIO.LOW)
            
                time.sleep(5)
                GPIO.output(led_pins, GPIO.LOW)
    except KeyboardInterrupt:
        print("Roger That. Exiting.")
    finally:
        GPIO.output(led_pins, GPIO.LOW)
        GPIO.cleanup()

