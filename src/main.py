import RPi.GPIO as GPIO
import json
import time

from nn import NeuralNetwork, np


class GPIOProcess:
    def __init__(self):
        self.led_pins = [12, 23, 32, 40]
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.led_pins, GPIO.OUT)

    def set_training_data(self):
        try:
            with open(".github/training/data.json", "r") as file:
                training_data = json.load(file)
        except FileNotFoundError:
            print("File not found!")

        self.training_data = training_data

    def train_model(self):
        for key in self.training_data:
            self.neural_network = NeuralNetwork(
                input_size=2, hidden_size=4, output_size=1
            )
            self.model = self.training_data[key]
            print(f"{key[9:]} model in progress.")

            for _ in range(10_000):
                for features in self.model:
                    inputs = np.array(features["input"])
                    expected_output = np.array(features["output"])
                    self.neural_network.train(
                        inputs, expected_output, learning_rate=0.1
                    )

    def set_model_output(self):
        outputs = []
        for features in self.model:
            inputs = np.array(features["input"])
            output = self.neural_network.feedforward(inputs)[0]
            print(f"{inputs} -> {output}")
            outputs.append(round(output))

        self.outputs = outputs

    def GPIO_output(self):
        for output, pin in zip(self.outputs, self.led_pins):
            GPIO.output(pin, GPIO.HIGH if output else GPIO.LOW)

        time.sleep(5)
        GPIO.output(self.led_pins, GPIO.LOW)
        print()

    def GPIO_clean(self):
        GPIO.output(self.led_pins, GPIO.LOW)
        GPIO.cleanup()


def main():
    process = GPIOProcess()
    process.set_training_data()
    try:
        process.train_model()
        process.set_model_output()
        process.GPIO_output()
    except KeyboardInterrupt:
        print("Roger That. Exiting.")
    finally:
        process.GPIO_clean()


if __name__ == "__main__":
    main()
