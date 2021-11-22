import random
import time

import numpy as np
import math


class MLP:
    def __init__(self, input_data_set, input_labels, train_data_set, train_labels, epochs=1000, l_rate=0.1, batch_size=0):
        self.number_of_parameters = len(input_data_set[0])

        self.hidden_layers_list = []
        self.layers_activation_functions = []

        self.input_data_set = input_data_set
        self.input_labels = input_labels

        self.train_data_set = train_data_set
        self.train_labels = train_labels

        self.matrices_with_weights = []
        self.biases = []

        self.mean_centre_of_distribution = 0
        self.standard_deviation = 0.1

        self.a_values = {}
        self.z_values = {}

        self.epochs = epochs

        self.l_rate = l_rate
        self.batch_size = batch_size

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

    def forward(self, x_train):
        self.a_values[0] = x_train

        for idx in range(len(self.get_layers_list()) - 1):
            # print('-------------------------')
            # print('Funkcja aktywacji: ', end='')
            # print(self.layers_activation_functions[idx])

            z = (self.matrices_with_weights[idx].dot(self.a_values[idx])) + self.biases[idx]
            self.z_values[idx + 1] = z
            if self.layers_activation_functions[idx] == 'sigm':
                self.a_values[idx + 1] = self.sigm(z)
            if self.layers_activation_functions[idx] == 'tanh':
                self.a_values[idx + 1] = self.tanh(z)
            if self.layers_activation_functions[idx] == 'relu':
                self.a_values[idx + 1] = self.relu(z)
            if self.layers_activation_functions[idx] == 'smax':
                self.a_values[idx + 1] = self.softmax(z)

            # print('Wyjście z warstwy: ', end='')
            # print(self.a_values[idx + 1])
        # print(self.a_values)
        return self.a_values[len(self.get_layers_list()) - 1]

    def backward(self, y_train, output):
        change_w = {}
        change_b = {}
        start_idx = len(self.get_layers_list()) - 1

        error = self.l_rate * (output - y_train) * self.softmax(self.z_values[start_idx], derivative=True)
        change_w[start_idx] = np.outer(error, self.a_values[start_idx - 1])
        change_b[start_idx] = error

        for idx in range(len(self.get_layers_list()) - 2, 0, -1):
            if self.layers_activation_functions[idx - 1] == 'sigm':
                error = np.dot(self.matrices_with_weights[idx].T, error) * self.sigm(self.z_values[idx], derivative=True)
            if self.layers_activation_functions[idx - 1] == 'tanh':
                error = np.dot(self.matrices_with_weights[idx].T, error) * self.tanh(self.z_values[idx], derivative=True)
            if self.layers_activation_functions[idx - 1] == 'relu':
                error = np.dot(self.matrices_with_weights[idx].T, error) * self.relu(self.z_values[idx], derivative=True)

            change_w[idx] = np.outer(error, self.a_values[idx - 1])
            change_b[idx] = error
        # print(change_b)
        return change_w, change_b

    def train(self):
        start_time = time.time()
        for i in range(self.epochs):

            batch_iterator = 0
            batch_change_w = []
            batch_change_b = []
            for x, y in zip(self.input_data_set, self.input_labels):
                output = self.forward(x)
                y_labels = np.zeros(len(output))
                y_labels[y] = 1
                change_w, change_b = self.backward(y_labels, output)
                if self.batch_size > 0:
                    if batch_iterator < self.batch_size:
                        batch_iterator += 1
                        batch_change_w.append(change_w)
                        batch_change_b.append(change_b)
                    else:
                        calculated_weights_change = batch_change_w[0]
                        calculated_biases_change = batch_change_b[0]
                        for change_idx in range(1, len(batch_change_w)):
                            calculated_weights_change = {key: np.add(calculated_weights_change[key], batch_change_w[change_idx][key]) for key in batch_change_w[change_idx].keys()}
                            calculated_biases_change = {key: np.add(calculated_biases_change[key], batch_change_b[change_idx][key]) for key in batch_change_b[change_idx].keys()}
                        calculated_weights_change = {key: calculated_weights_change[key] / len(batch_change_w) for key in calculated_weights_change.keys()}
                        calculated_biases_change = {key: calculated_biases_change[key] / len(batch_change_w) for key in calculated_biases_change.keys()}

                        self.update_weights(calculated_weights_change)
                        self.update_biases(calculated_biases_change)

                        batch_iterator = 0
                        batch_change_w = []
                        batch_change_b = []

                else:
                    self.update_weights(change_w)
                    self.update_biases(change_b)

            accuracy = self.test_accuracy(self.train_data_set, self.train_labels)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2}'.format(
                i + 1, time.time() - start_time, accuracy
            ))

    def update_weights(self, change_w):
        for i in range(len(self.hidden_layers_list) + 1):
            # self.matrices_with_weights[i] -= self.l_rate * change_w[i + 1]
            self.matrices_with_weights[i] -= change_w[i + 1]

    def update_biases(self, change_b):
        for i in range(len(self.hidden_layers_list) + 1):
            # self.biases[i] -= self.l_rate * change_b[i + 1]
            self.biases[i] -= change_b[i + 1]

    def test_accuracy(self, x_val, y_val):
        predictions = []

        for x, y in zip(x_val, y_val):
            output = self.forward(x)
            pred = np.argmax(output)
            predictions.append(pred == y)
        # print(np.mean(predictions))
        return np.mean(predictions)

    def sigm(self, x, derivative=False):
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))

    def tanh(self, x, derivative=False):
        if derivative:
            return 1 - np.tanh(x)**2
        return np.tanh(x)

    def relu(self, x, derivative=False):
        if derivative:
            return np.maximum(np.minimum(x, 1), 0)
        return np.maximum(0, x)

    def softmax(self, x, derivative=False):
        exps = np.exp(x - x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)
