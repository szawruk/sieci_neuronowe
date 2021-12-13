from copy import deepcopy

from keras.datasets import mnist

import numpy as np
# Press the green button in the gutter to run the script.
from MLP import MLP

if __name__ == '__main__':
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = train_X.reshape((train_X.shape[0], 28 * 28)) / 255
    test_X = test_X.reshape((test_X.shape[0], 28 * 28)) / 255
    
    train_size = 20000
    test_size = 2000

    # print('standard')
    # mlp = MLP(train_X[0:train_size], train_y[0:train_size], test_X[0:test_size], test_y[0:test_size], epochs=100, l_rate=0.01, batch_size=0, deviation=0.1, l_rate_optimizer='standard')
    # mlp.add_layer(32, 'relu')
    # mlp.finalize_model()
    #
    # matrices_with_weights = deepcopy(mlp.matrices_with_weights)
    # matrices_with_previous_weights_updates = deepcopy(mlp.matrices_with_previous_weights_updates)
    # matrices_with_predicted_weights = deepcopy(mlp.matrices_with_predicted_weights)
    # gradient_sum = deepcopy(mlp.gradient_sum)
    #
    # biases = deepcopy(mlp.biases)
    # biases_with_previous_updates = deepcopy(mlp.biases_with_previous_updates)
    # biases_predicted = deepcopy(mlp.biases_predicted)
    # gradient_sum_biases = deepcopy(mlp.gradient_sum_biases)
    #
    # mlp.train()
    #
    # print()
    # print('momentum')
    # mlp = MLP(train_X[0:train_size], train_y[0:train_size], test_X[0:test_size], test_y[0:test_size], epochs=100, l_rate=0.01, batch_size=0, deviation=0.1, l_rate_optimizer='momentum')
    # mlp.add_layer(32, 'relu')
    # mlp.finalize_model()
    #
    # mlp.matrices_with_weights = deepcopy(matrices_with_weights)
    # mlp.matrices_with_previous_weights_updates = deepcopy(matrices_with_previous_weights_updates)
    # mlp.matrices_with_predicted_weights = deepcopy(matrices_with_predicted_weights)
    # mlp.gradient_sum = deepcopy(gradient_sum)
    #
    # mlp.biases = deepcopy(biases)
    # mlp.biases_with_previous_updates = deepcopy(biases_with_previous_updates)
    # mlp.biases_predicted = deepcopy(biases_predicted)
    # mlp.gradient_sum_biases = deepcopy(gradient_sum_biases)
    #
    # mlp.train()
    #
    # print()
    # print('nesterov')
    # mlp = MLP(train_X[0:train_size], train_y[0:train_size], test_X[0:test_size], test_y[0:test_size], epochs=100, l_rate=0.01, batch_size=0, deviation=0.1, l_rate_optimizer='nesterov')
    # mlp.add_layer(32, 'relu')
    # mlp.finalize_model()
    #
    # mlp.matrices_with_weights = deepcopy(matrices_with_weights)
    # mlp.matrices_with_previous_weights_updates = deepcopy(matrices_with_previous_weights_updates)
    # mlp.matrices_with_predicted_weights = deepcopy(matrices_with_predicted_weights)
    # mlp.gradient_sum = deepcopy(gradient_sum)
    #
    # mlp.biases = deepcopy(biases)
    # mlp.biases_with_previous_updates = deepcopy(biases_with_previous_updates)
    # mlp.biases_predicted = deepcopy(biases_predicted)
    # mlp.gradient_sum_biases = deepcopy(gradient_sum_biases)
    #
    # mlp.train()
    # print()
    # print('adagrad')
    # mlp = MLP(train_X[0:train_size], train_y[0:train_size], test_X[0:test_size], test_y[0:test_size], epochs=100, l_rate=0.01, batch_size=0, deviation=0.1, l_rate_optimizer='adagard')
    # mlp.add_layer(32, 'relu')
    # mlp.finalize_model()
    #
    # mlp.matrices_with_weights = deepcopy(matrices_with_weights)
    # mlp.matrices_with_previous_weights_updates = deepcopy(matrices_with_previous_weights_updates)
    # mlp.matrices_with_predicted_weights = deepcopy(matrices_with_predicted_weights)
    # mlp.gradient_sum = deepcopy(gradient_sum)
    #
    # mlp.biases = deepcopy(biases)
    # mlp.biases_with_previous_updates = deepcopy(biases_with_previous_updates)
    # mlp.biases_predicted = deepcopy(biases_predicted)
    # mlp.gradient_sum_biases = deepcopy(gradient_sum_biases)
    #
    # mlp.train()

    print()
    print('none')
    mlp = MLP(train_X[0:train_size], train_y[0:train_size], test_X[0:test_size], test_y[0:test_size], epochs=100, l_rate=0.01, batch_size=0, deviation=0.1, l_rate_optimizer='standard', weights_optimizer='none')
    mlp.add_layer(32, 'relu')
    mlp.finalize_model()
    mlp.train()

    print()
    print('xavier')
    mlp = MLP(train_X[0:train_size], train_y[0:train_size], test_X[0:test_size], test_y[0:test_size], epochs=100, l_rate=0.01, batch_size=0, deviation=0.1, l_rate_optimizer='standard', weights_optimizer='xavier')
    mlp.add_layer(32, 'relu')
    mlp.finalize_model()
    mlp.train()

    print()
    print('he')
    mlp = MLP(train_X[0:train_size], train_y[0:train_size], test_X[0:test_size], test_y[0:test_size], epochs=100, l_rate=0.01, batch_size=0, deviation=0.1, l_rate_optimizer='standard', weights_optimizer='he')
    mlp.add_layer(32, 'relu')
    mlp.finalize_model()
    mlp.train()








