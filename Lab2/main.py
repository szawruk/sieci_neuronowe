from keras.datasets import mnist

import numpy as np
# Press the green button in the gutter to run the script.
from MLP import MLP

if __name__ == '__main__':
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = train_X.reshape((train_X.shape[0], 28 * 28)) / 255
    test_X = test_X.reshape((test_X.shape[0], 28 * 28)) / 255
    
    train_size = 15000
    test_size = 1500

    # mlp = MLP(train_X[0:train_size], train_y[0:train_size], test_X[0:test_size], test_y[0:test_size], epochs=100, l_rate=0.01, batch_size=0)
    # mlp.add_layer(256, 'sigm')
    # print('liczba neuronow: ', 256)
    # mlp.finalize_model()
    # mlp.train()

    # --------------------BADANIE WSPOLCZYNNIKOW UCZENIA

    # mlp = MLP(train_X[0:train_size], train_y[0:train_size], test_X[0:test_size], test_y[0:test_size], epochs=100, l_rate=0.00001, batch_size=0)
    # mlp.add_layer(32, 'sigm')
    # # mlp.add_layer(32, 'sigm')
    # print('learning rate;{0}', mlp.l_rate)
    # mlp.finalize_model()
    # mlp.train()
    #
    # mlp = MLP(train_X[0:train_size], train_y[0:train_size], test_X[0:test_size], test_y[0:test_size], epochs=100, l_rate=0.0001, batch_size=0)
    # mlp.add_layer(32, 'sigm')
    # # mlp.add_layer(32, 'sigm')
    # print('learning rate;{0}', mlp.l_rate)
    # mlp.finalize_model()
    # mlp.train()
    #
    # mlp = MLP(train_X[0:train_size], train_y[0:train_size], test_X[0:test_size], test_y[0:test_size], epochs=100, l_rate=0.001, batch_size=0)
    # mlp.add_layer(32, 'sigm')
    # # mlp.add_layer(32, 'sigm')
    # print('learning rate;{0}', mlp.l_rate)
    # mlp.finalize_model()
    # mlp.train()
    #
    # mlp = MLP(train_X[0:train_size], train_y[0:train_size], test_X[0:test_size], test_y[0:test_size], epochs=100, l_rate=0.01, batch_size=0)
    # mlp.add_layer(32, 'sigm')
    # # mlp.add_layer(32, 'sigm')
    # print('learning rate;{0}', mlp.l_rate)
    # mlp.finalize_model()
    # mlp.train()
    #
    # mlp = MLP(train_X[0:train_size], train_y[0:train_size], test_X[0:test_size], test_y[0:test_size], epochs=100, l_rate=0.1, batch_size=0)
    # mlp.add_layer(32, 'sigm')
    # # mlp.add_layer(32, 'sigm')
    # print('learning rate;{0}', mlp.l_rate)
    # mlp.finalize_model()
    # mlp.train()


    # # --------------------BADANIE WIELKOSCI BATCHA
    #
    # mlp = MLP(train_X[0:train_size], train_y[0:train_size], test_X[0:test_size], test_y[0:test_size], epochs=100, l_rate=0.1, batch_size=10)
    # mlp.add_layer(32, 'sigm')
    # print('batch size: ', mlp.batch_size)
    # mlp.finalize_model()
    # mlp.train()
    #
    # mlp = MLP(train_X[0:train_size], train_y[0:train_size], test_X[0:test_size], test_y[0:test_size], epochs=100, l_rate=0.1, batch_size=30)
    # mlp.add_layer(32, 'sigm')
    # print('batch size: ', mlp.batch_size)
    # mlp.finalize_model()
    # mlp.train()
    #
    # mlp = MLP(train_X[0:train_size], train_y[0:train_size], test_X[0:test_size], test_y[0:test_size], epochs=100, l_rate=0.1, batch_size=80)
    # mlp.add_layer(32, 'sigm')
    # print('batch size: ', mlp.batch_size)
    # mlp.finalize_model()
    # mlp.train()
    #
    # mlp = MLP(train_X[0:train_size], train_y[0:train_size], test_X[0:test_size], test_y[0:test_size], epochs=100, l_rate=0.1, batch_size=150)
    # mlp.add_layer(32, 'sigm')
    # print('batch size: ', mlp.batch_size)
    # mlp.finalize_model()
    # mlp.train()
    #
    # mlp = MLP(train_X[0:train_size], train_y[0:train_size], test_X[0:test_size], test_y[0:test_size], epochs=100, l_rate=0.1, batch_size=250)
    # mlp.add_layer(32, 'sigm')
    # print('batch size: ', mlp.batch_size)
    # mlp.finalize_model()
    # mlp.train()


    # --------------------BADANIE ROZNE ODCHYLENIE STANDARDOWE LOSOWANIE WAG

    # mlp = MLP(train_X[0:train_size], train_y[0:train_size], test_X[0:test_size], test_y[0:test_size], epochs=100, l_rate=0.01, batch_size=0, deviation=0.1)
    # mlp.add_layer(32, 'sigm')
    # print('odchylenie standardowe: ', mlp.standard_deviation)
    # mlp.finalize_model()
    # mlp.train()
    #
    # mlp = MLP(train_X[0:train_size], train_y[0:train_size], test_X[0:test_size], test_y[0:test_size], epochs=100, l_rate=0.01, batch_size=0, deviation=0.3)
    # mlp.add_layer(32, 'sigm')
    # print('odchylenie standardowe: ', mlp.standard_deviation)
    # mlp.finalize_model()
    # mlp.train()
    #
    # mlp = MLP(train_X[0:train_size], train_y[0:train_size], test_X[0:test_size], test_y[0:test_size], epochs=100, l_rate=0.01, batch_size=0, deviation=0.6)
    # mlp.add_layer(32, 'sigm')
    # print('odchylenie standardowe: ', mlp.standard_deviation)
    # mlp.finalize_model()
    # mlp.train()
    #
    # mlp = MLP(train_X[0:train_size], train_y[0:train_size], test_X[0:test_size], test_y[0:test_size], epochs=100, l_rate=0.01, batch_size=0, deviation=0.8)
    # mlp.add_layer(32, 'sigm')
    # print('odchylenie standardowe: ', mlp.standard_deviation)
    # mlp.finalize_model()
    # mlp.train()
    #
    # mlp = MLP(train_X[0:train_size], train_y[0:train_size], test_X[0:test_size], test_y[0:test_size], epochs=100, l_rate=0.01, batch_size=0, deviation=1)
    # mlp.add_layer(32, 'sigm')
    # print('odchylenie standardowe: ', mlp.standard_deviation)
    # mlp.finalize_model()
    # mlp.train()

    # --------------------BADANIE ROZNE FUNKCJE AKTYWACJI

    mlp = MLP(train_X[0:train_size], train_y[0:train_size], test_X[0:test_size], test_y[0:test_size], epochs=100, l_rate=0.01, batch_size=0)
    mlp.add_layer(64, 'sigm')
    print('sigm')
    mlp.finalize_model()
    mlp.train()

    mlp = MLP(train_X[0:train_size], train_y[0:train_size], test_X[0:test_size], test_y[0:test_size], epochs=100, l_rate=0.01, batch_size=0)
    mlp.add_layer(64, 'tanh')
    print('tanh')
    mlp.finalize_model()
    mlp.train()

    mlp = MLP(train_X[0:train_size], train_y[0:train_size], test_X[0:test_size], test_y[0:test_size], epochs=100, l_rate=0.01, batch_size=0)
    mlp.add_layer(64, 'relu')
    print('relu')
    mlp.finalize_model()
    mlp.train()









