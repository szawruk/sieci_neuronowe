from keras.datasets import mnist
from MLP import MLP

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    train_X = train_X.reshape((train_X.shape[0], 28 * 28))
    # print('X_train: ' + str(train_X.shape))

    mlp = MLP(train_X, train_y)
    mlp.add_layer(20, 'sigm')
    mlp.add_layer(18, 'sigm')
    mlp.add_layer(14, 'relu')
    mlp.finalize_model()
    mlp.calculate_output()

