from keras.datasets import mnist

import numpy as np
# Press the green button in the gutter to run the script.
from Lab2.MLP import MLP

if __name__ == '__main__':
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = train_X.reshape((train_X.shape[0], 28 * 28)) / 255
    test_X = test_X.reshape((test_X.shape[0], 28 * 28)) / 255

    # print('X_train: ' + str(train_X.shape))

    mlp = MLP(train_X[0:2000], train_y[0:2000], test_X[0:1000], test_y[0:1000], epochs=1000, l_rate=0.01, batch_size=0)
    mlp.add_layer(64, 'tanh')
    mlp.add_layer(32, 'tanh')
    # mlp.add_layer(16, 'relu')
    mlp.finalize_model()
    mlp.train()
    # out = mlp.forward(mlp.input_data_set[0])
    # mlp.backward(mlp.input_labels[0], out)





