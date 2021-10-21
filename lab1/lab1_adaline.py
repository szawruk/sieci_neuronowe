import random

learning_parameter = 0.01
error_threshold = 0.4


def calc_output_value(input_vector, weights):
    sum_z = 0

    for i in range(len(input_vector)):
        sum_z += input_vector[i] * weights[i]

    if sum_z > 0:
        return 1
    else:
        return -1


def adaline_get_error(weights, input_values, real_y):
    return real_y - sum([a * b for a, b in zip(weights, input_values)])


def adaline_change_weights(weights, input_values, error, learning_param):
    # print(weights)
    # print()
    for i in range(len(weights)):
        weights[i] += (2 * learning_param * error * input_values[i])


def train_perceptron(case_name, training_set, y, weights, threshold, learning_param):
    epochs = 0
    was_error = True

    while was_error:
        was_error = False
        square_error_sum = 0
        for i in range(len(training_set)):
            # perceptron_output = calc_output_value(training_set[i], weights)

            error = adaline_get_error(weights, training_set[i], y[i])
            square_error_sum += error ** 2

            adaline_change_weights(weights, training_set[i], error, learning_param)

        if (square_error_sum / len(training_set)) > threshold:
            was_error = True

        epochs += 1
        if epochs > 50000:
            print('PRZEKROCZONO 50000 EPOK! ', end='')
            break

    print(case_name + ' liczba epok: ' + str(epochs) + ' wartosc progu bledu: ' + str(
        threshold) + ' learning param: ' + str(learning_param) + '  ', end='')
    print("koncowe wagi: ", end='')
    print(weights)
    return epochs


def weights_test(training_set, y, weights_range, learning_param):
    epochs = 0
    for i in range(10):
        weights = [random.random() * weights_range * 2 - 1, random.random() * weights_range * 2 - 1, random.random() * weights_range * 2 - 1]
        epochs += train_perceptron('Adaline zad 1', training_set, y, weights, 0.5, learning_param)
    print('EPOK SREDNIO:' + str(epochs / 10))

def main():
    learning_parameter = 0.01
    weights = [0.1, -0.2, 0.1]
    training_set = [[1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1], [1, 0.9, 0.8], [1, 0.96, 0.97]]
    y = [-1, -1, -1, 1, 1, 1]

    for i in [1, 0.8, 0.6, 0.4, 0.2]:
        weights_test(training_set, y, i, learning_parameter)

    print('------------------------')

    for i in [0.00001, 0.0001, 0.001, 0.01, 0.1]:
        weights = [0.1, -0.2, 0.1]
        training_set = [[1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1], [1, 0.9, 0.8], [1, 0.96, 0.97]]
        y = [-1, -1, -1, 1, 1, 1]
        train_perceptron('Adaline zad 2', training_set, y, weights, 0.5, i)

    print('------------------------')

    thresholds = [0.25, 0.3, 0.35, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 5]
    learning_parameter = 0.0001

    for i in thresholds:
        weights = [0.1, -0.2, 0.1]
        training_set = [[1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1], [1, 0.9, 0.8], [1, 0.96, 0.97]]
        y = [-1, -1, -1, 1, 1, 1]
        train_perceptron('Adaline zad 3', training_set, y, weights, i, learning_parameter)


if __name__ == "__main__":
    main()


