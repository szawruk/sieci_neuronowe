from keras.callbacks import EarlyStopping
from tensorflow import keras


def getNormalNeuralNetwork(layers_number=2):
    neural_network = keras.models.Sequential()
    neural_network.add(keras.layers.Flatten(input_shape=(28, 28)))

    for i in range(layers_number):
        neural_network.add(keras.layers.Dense(32, activation='sigmoid'))

    neural_network.add(keras.layers.Dense(10, activation='softmax'))
    neural_network.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return neural_network


def getCNNNetwork(layers_number=2, kernel_size=4, add_pooling_layer=False, pool_size=2):
    neural_network = keras.models.Sequential()
    neural_network.add(keras.layers.Conv2D(32, kernel_size=kernel_size, activation='sigmoid', input_shape=(28, 28, 1)))
    if add_pooling_layer:
        neural_network.add(keras.layers.MaxPooling2D(pool_size=pool_size, padding='same'))

    for i in range(layers_number - 1):
        neural_network.add(keras.layers.Conv2D(32, kernel_size=kernel_size, activation='sigmoid'))
        if add_pooling_layer:
            neural_network.add(keras.layers.MaxPooling2D(pool_size=pool_size, padding='same'))

    neural_network.add(keras.layers.Flatten())
    neural_network.add(keras.layers.Dense(10, activation='softmax'))

    neural_network.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    return neural_network
