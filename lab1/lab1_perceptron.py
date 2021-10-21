import numpy as np
import random
learning_parameter = 0.1


def calc_output_value(input_vector, weights, threshold):
    if len(input_vector) != len(weights):
        print('input arrays have invalid input!')
        return

    sum_z = 0

    for i in range(len(input_vector)):
        sum_z += input_vector[i] * weights[i]

    if sum_z > threshold:
        return 1
    else:
        return 0


def change_weights(weights, input_values, error, learning_param):
    # print(weights)
    # print()
    for i in range(len(weights)):
        weights[i] += learning_param * error * input_values[i]


def train_perceptron(case_name, training_set, y, weights, threshold, learning_param):
    was_error = True
    epochs = 0

    while was_error:
        was_error = False
        for i in range(len(training_set)):
            perceptron_output = calc_output_value(training_set[i], weights, threshold)
            error_value = y[i] - perceptron_output
            print(error_value)
            if error_value != 0:
                was_error = True
            change_weights(weights, training_set[i], error_value, learning_param)

        epochs += 1
        if epochs > 50000:
            print('PRZEKROCZONO 50000 EPOK! ', end='')
            break

    print(case_name + ' liczba epok: ' + str(epochs) + ' wartosc progu: ' + str(threshold) + ' learning param: ' + str(learning_param) +'  ', end='')
    print("koncowe wagi: ", end='')
    print(weights)
    return epochs


def weights_test(training_set, y, weights_range, learning_param):
    epochs = 0
    for i in range(10):
        weights = [random.random() * weights_range * 2 - 1, random.random() * weights_range * 2 - 1]
        epochs += train_perceptron('Perceptron zad 2', training_set, y, weights, 0.5, learning_param)
    print('EPOK SREDNIO:' + str(epochs / 10))


def learning_parameter_test(training_set, y, weights, learning_param):
    epochs = 0
    for i in range(10):
        epochs += train_perceptron('Perceptron zad 2', training_set, y, weights, 0.5, learning_param)
    print('EPOK SREDNIO:' + str(epochs / 10))



def main():
    weights = [0.2, 0.7]
    learning_parameter = 0.1

    training_set = [[0, 0], [0, 1], [1, 0], [1, 1], [0.9, 0.8], [0.96, 0.97]]
    y = [0, 0, 0, 1, 1, 1]

    thresholds = [0.1, 0.2, 0.3, 0.5, 0.9, 5, 10, 20, 50, 120]

    for i in thresholds:
        train_perceptron('Perceptron zad 1', training_set, y, weights, i, learning_parameter)

    print('------------------------')

    for i in [1, 0.8, 0.6, 0.4, 0.2]:
        weights_test(training_set, y, i, learning_parameter)

    print('------------------------')

    for i in [0.00001, 0.0001, 0.001, 0.01, 0.1]:
        weights = [0.2, 0.7]
        training_set = [[0, 0], [0, 1], [1, 0], [1, 1], [0.9, 0.8], [0.96, 0.97]]
        y = [0, 0, 0, 1, 1, 1]
        train_perceptron('Perceptron zad 3', training_set, y, weights, 0.5, i)

    print('------------------------')

    # funkcja bipolarna, potrzebne zminay wprowadzone ręcznie w czasie odpalenia kodu
    # weights = [0.2, 0.7]
    # training_set = [[-1, -1], [-1, 1], [1, -1], [1, 1], [0.9, 0.8], [0.96, 0.97]]
    # y = [-1, -1, -1, 1, 1, 1]
    #
    # train_perceptron('Perceptron zad 4', training_set, y, weights, 0.9, 0.0001)

    weights = [0.2, 0.7]
    training_set = [[0, 0], [0, 1], [1, 0], [1, 1], [0.9, 0.8], [0.96, 0.97]]
    y = [0, 0, 0, 1, 1, 1]
    train_perceptron('Perceptron zad 4', training_set, y, weights, 0.9, 0.0001)

if __name__ == "__main__":
    main()
