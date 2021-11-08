import random

import numpy as np
import math


class MLP:
    def __init__(self, input_data_set, input_labels):
        self.number_of_parameters = len(input_data_set[0])

        self.hidden_layers_list = []
        self.layers_activation_functions = []

        self.input_data_set = input_data_set / 255
        self.input_labels = input_labels

        self.matrices_with_weights = []
        self.biases = []

        self.mean_centre_of_distribution = 0
        self.standard_deviation = 1

    def calculate_weights_matrixes(self):
        for x in range(len(self.hidden_layers_list) + 1):
            self.matrices_with_weights.append(
                np.random.normal(self.mean_centre_of_distribution, self.standard_deviation,
                                 (self.get_layers_list()[x + 1], self.get_layers_list()[x])))
            self.biases.append([random.random() for i in range(self.get_layers_list()[x + 1])])

    # pierwsza warstwa ma tyle neuronów ile parametrów wejsciowych
    # ostatnia warstwa ma 10 neuronów, ponieważ tyle mamy kategorii w tym problemie (cyfry)
    def get_layers_list(self):
        return [self.number_of_parameters] + self.hidden_layers_list + [10]

    def add_layer(self, number_of_neurons, activation_function):
        if activation_function not in ['sigm', 'tanh', 'relu']:
            print("Bad activation function")
            return

        self.hidden_layers_list.append(number_of_neurons)
        self.layers_activation_functions.append(activation_function)

    def finalize_model(self):
        self.calculate_weights_matrixes()
        self.layers_activation_functions.append('smax')

    def calculate_output(self):
        actual_values_vector = self.input_data_set[0]

        for idx in range(len(self.get_layers_list()) - 1):
            print('-------------------------')
            print('Funkcja aktywacji: ', end='')
            print(self.layers_activation_functions[idx])

            z = (self.matrices_with_weights[idx].dot(actual_values_vector)) + self.biases[idx]
            if self.layers_activation_functions[idx] == 'sigm':
                actual_values_vector = self.sigm(z)
            if self.layers_activation_functions[idx] == 'tanh':
                actual_values_vector = self.tanh(z)
            if self.layers_activation_functions[idx] == 'relu':
                actual_values_vector = self.relu(z)
            if self.layers_activation_functions[idx] == 'smax':
                actual_values_vector = self.softmax(z)

            print('Wyjście z warstwy: ', end='')
            print(actual_values_vector)

    def sigm(self, x):
        return x / (1 + math.e ** (-1))

    def tanh(self, x):
        return (2 / (1 + math.e ** (-2 * x))) - 1

    def relu(self, x):
        for i in range(len(x)):
            if x[i] < 0:
                x[i] = 0
        return x

    def softmax(self, input_vector):
        exp_sum = 0
        for i in input_vector:
            exp_sum += math.exp(i)

        for i in range(len(input_vector)):
            input_vector[i] = math.exp(input_vector[i]) / exp_sum

        return input_vector
