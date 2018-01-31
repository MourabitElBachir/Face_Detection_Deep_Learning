import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from code.preparing_data import *
import os


def training(nb_epoch=5):

    # Loading training and test data  :
    if os.path.exists(TRAIN_DB):
        train_data = np.load(TRAIN_DB)
        print("Load Train data")
    else:
        train_data = create_train_data()

    if os.path.exists(TEST_DB):
        test_data = np.load(TEST_DB)
        print("Load Test data")
    else:
        test_data = process_test_data()

    train = train_data
    test = test_data

    X = np.array([i[0] for i in train]).reshape(-1, IMG_SIZE_WIDTH, IMG_SIZE_HEIGHT, 1)
    Y = [i[1] for i in train]

    test_x = np.array([i[0] for i in test]).reshape(-1, IMG_SIZE_WIDTH, IMG_SIZE_HEIGHT, 1)
    test_y = [i[1] for i in test]

    # Define and use the neural network :

    convnet = input_data(shape=[None, IMG_SIZE_WIDTH, IMG_SIZE_HEIGHT, 1], name='input')

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 128, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)

    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)

    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    model.fit({'input': X}, {'targets': Y}, n_epoch=nb_epoch, validation_set=({'input': test_x}, {'targets': test_y}),
              snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

    model.save(MODEL_LOCATION)


if __name__ == "__main__":

    training(20)
