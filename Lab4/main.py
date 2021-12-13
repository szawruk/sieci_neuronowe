import time
from copy import deepcopy

from tensorflow.keras.callbacks import EarlyStopping, Callback
from keras.datasets import mnist

import numpy as np

from Lab4 import networkGenerator

(train_X, train_y), (test_X, test_y) = mnist.load_data()

train_X = train_X.reshape((train_X.shape[0], 28, 28, 1))
test_X = test_X.reshape((test_X.shape[0], 28, 28, 1))

train_size = 30000
test_size = 5000

epochs = 6
iterations = 5

# BADANIE LICZBY WARSTW KONWOLUCYJNYCH
def research1():
    params = [1, 2, 3, 4]
    accuracy_list_result = []
    time_list_result = []

    for param in params:

        accuracy_list = np.zeros(epochs)
        time_list = np.zeros(epochs)

        for i in range(iterations):
            print(f"parametr: {param}, iteracja: {i}")

            start = time.time()

            model = networkGenerator.getCNNNetwork(layers_number=param, kernel_size=4, add_pooling_layer=False, pool_size=2)
            history = model.fit(train_X[0:train_size], train_y[0:train_size], validation_split=0.2, epochs=epochs)

            stop = time.time()
            time_per_epoch = (stop - start) / epochs

            print(time_per_epoch)

            for idx in range(epochs):
                accuracy_list[idx] += history.history['val_accuracy'][idx]
                time_list[idx] += time_per_epoch * (idx + 1)

        for idx in range(epochs):
            accuracy_list[idx] /= iterations
            time_list[idx] /= iterations

        accuracy_list_result.append(accuracy_list)
        time_list_result.append(time_list)

    print('badanie liczby warstw konwolucyjnych-------------------------------------------------')
    for i in range(len(accuracy_list_result)):
        for y in range(len(accuracy_list_result[i])):
            print('{0};{1};{2};{3}'.format(
                 params[i], y + 1, str(round(accuracy_list_result[i][y], 4)).replace('.', ','), str(round(time_list_result[i][y], 2)).replace('.', ',')
            ))
        print()


# BADANIE ROZMIARU KERNEL
def research2():
    params = [2, 3, 4, 5, 6]
    accuracy_list_result = []
    time_list_result = []

    for param in params:

        accuracy_list = np.zeros(epochs)
        time_list = np.zeros(epochs)

        for i in range(iterations):
            print(f"parametr: {param}, iteracja: {i}")

            start = time.time()

            model = networkGenerator.getCNNNetwork(layers_number=2, kernel_size=param, add_pooling_layer=False, pool_size=2)
            history = model.fit(train_X[0:train_size], train_y[0:train_size], validation_split=0.2, epochs=epochs)

            stop = time.time()
            time_per_epoch = (stop - start) / epochs

            print(time_per_epoch)

            for idx in range(epochs):
                accuracy_list[idx] += history.history['val_accuracy'][idx]
                time_list[idx] += time_per_epoch * (idx + 1)

        for idx in range(epochs):
            accuracy_list[idx] /= iterations
            time_list[idx] /= iterations

        accuracy_list_result.append(accuracy_list)
        time_list_result.append(time_list)

    print('badanie wielkości kernerla-------------------------------------------------')
    for i in range(len(accuracy_list_result)):
        for y in range(len(accuracy_list_result[i])):
            print('{0};{1};{2};{3}'.format(
                 params[i], y + 1, str(round(accuracy_list_result[i][y], 4)).replace('.', ','), str(round(time_list_result[i][y], 2)).replace('.', ',')
            ))
        print()


# BADANIE POOLING I BEZ POOLING
def research3():
    params = [True, False]
    accuracy_list_result = []
    time_list_result = []

    for param in params:

        accuracy_list = np.zeros(epochs)
        time_list = np.zeros(epochs)

        for i in range(iterations):
            print(f"parametr: {param}, iteracja: {i}")

            start = time.time()

            model = networkGenerator.getCNNNetwork(layers_number=2, kernel_size=4, add_pooling_layer=param, pool_size=2)
            history = model.fit(train_X[0:train_size], train_y[0:train_size], validation_split=0.2, epochs=epochs)

            stop = time.time()
            time_per_epoch = (stop - start) / epochs

            print(time_per_epoch)

            for idx in range(epochs):
                accuracy_list[idx] += history.history['val_accuracy'][idx]
                time_list[idx] += time_per_epoch * (idx + 1)

        for idx in range(epochs):
            accuracy_list[idx] /= iterations
            time_list[idx] /= iterations

        accuracy_list_result.append(accuracy_list)
        time_list_result.append(time_list)

    print('badanie pooling i bez pooling-------------------------------------------------')
    for i in range(len(accuracy_list_result)):
        for y in range(len(accuracy_list_result[i])):
            print('{0};{1};{2};{3}'.format(
                params[i], y + 1, str(round(accuracy_list_result[i][y], 4)).replace('.', ','), str(round(time_list_result[i][y], 2)).replace('.', ',')
            ))
        print()


# BADANIE POOLING SIZE
def research4():
    params = [2,3,4]
    accuracy_list_result = []
    time_list_result = []

    for param in params:

        accuracy_list = np.zeros(epochs)
        time_list = np.zeros(epochs)

        for i in range(iterations):
            print(f"parametr: {param}, iteracja: {i}")

            start = time.time()

            model = networkGenerator.getCNNNetwork(layers_number=2, kernel_size=4, add_pooling_layer=True, pool_size=param)
            history = model.fit(train_X[0:train_size], train_y[0:train_size], validation_split=0.2, epochs=epochs)

            stop = time.time()
            time_per_epoch = (stop - start) / epochs

            print(time_per_epoch)

            for idx in range(epochs):
                accuracy_list[idx] += history.history['val_accuracy'][idx]
                time_list[idx] += time_per_epoch * (idx + 1)

        for idx in range(epochs):
            accuracy_list[idx] /= iterations
            time_list[idx] /= iterations

        accuracy_list_result.append(accuracy_list)
        time_list_result.append(time_list)

    print('badanie różne pool size average pooling-------------------------------------------------')
    for i in range(len(accuracy_list_result)):
        for y in range(len(accuracy_list_result[i])):
            print('{0};{1};{2};{3}'.format(
                params[i], y + 1, str(round(accuracy_list_result[i][y], 4)).replace('.', ','), str(round(time_list_result[i][y], 2)).replace('.', ',')
            ))
        print()


# BADANIE cnn vs normal
def research5():
    params = [1]
    accuracy_list_result = []
    time_list_result = []

    epochs = 10

    for param in params:

        accuracy_list = np.zeros(epochs)
        time_list = np.zeros(epochs)

        for i in range(iterations):
            print(f"parametr: {param}, iteracja: {i}")

            start = time.time()

            model = networkGenerator.getCNNNetwork(layers_number=2, kernel_size=4, add_pooling_layer=False, pool_size=2)
            history = model.fit(train_X[0:train_size], train_y[0:train_size], validation_split=0.2, epochs=epochs)

            stop = time.time()
            time_per_epoch = (stop - start) / epochs

            print(time_per_epoch)

            for idx in range(epochs):
                accuracy_list[idx] += history.history['val_accuracy'][idx]
                time_list[idx] += time_per_epoch * (idx + 1)

        for idx in range(epochs):
            accuracy_list[idx] /= iterations
            time_list[idx] /= iterations

        accuracy_list_result.append(accuracy_list)
        time_list_result.append(time_list)

    print('badanie cnn-------------------------------------------------')
    for i in range(len(accuracy_list_result)):
        for y in range(len(accuracy_list_result[i])):
            print('{0};{1};{2};{3}'.format(
                params[i], y + 1, str(round(accuracy_list_result[i][y], 4)).replace('.', ','), str(round(time_list_result[i][y], 2)).replace('.', ',')
            ))
        print()

    accuracy_list_result = []
    time_list_result = []

    for param in params:

        accuracy_list = np.zeros(epochs)
        time_list = np.zeros(epochs)

        for i in range(iterations):
            print(f"parametr: {param}, iteracja: {i}")

            start = time.time()

            model = networkGenerator.getNormalNeuralNetwork(layers_number=3)
            history = model.fit(train_X[0:train_size], train_y[0:train_size], validation_split=0.2, epochs=epochs)

            stop = time.time()
            time_per_epoch = (stop - start) / epochs

            print(time_per_epoch)

            for idx in range(epochs):
                accuracy_list[idx] += history.history['val_accuracy'][idx]
                time_list[idx] += time_per_epoch * (idx + 1)

        for idx in range(epochs):
            accuracy_list[idx] /= iterations
            time_list[idx] /= iterations

        accuracy_list_result.append(accuracy_list)
        time_list_result.append(time_list)

    print('badanie normal-------------------------------------------------')
    for i in range(len(accuracy_list_result)):
        for y in range(len(accuracy_list_result[i])):
            print('{0};{1};{2};{3}'.format(
                params[i], y + 1, str(round(accuracy_list_result[i][y], 4)).replace('.', ','), str(round(time_list_result[i][y], 2)).replace('.', ',')
            ))
        print()


if __name__ == '__main__':
    # research1()
    research2()
    # research3()
    # research4()
    # research5()


def main():
    model = networkGenerator.getCNNNetwork()
    # model = networkGenerator.getNormalNeuralNetwork()
    model.fit(train_X[0:train_size], train_y[0:train_size], epochs=epochs)
    acc = model.evaluate(test_X[0:test_size], test_y[0:test_size], verbose=2)

    for i in range(len(acc)):
        print('{0};{1}'.format(
            i + 1, str(round(acc[i] * 100, 2)).replace('.', ',')
        ))

    print(f"Epochs: {epochs}, acc: {round(acc[len(acc) - 1] * 100, 2)}%!")

    print("End!")
