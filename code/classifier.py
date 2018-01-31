import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import os
import cv2
import tensorflow as tf
from code.env_variables import *


def own_model():

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

    return convnet


def vgg16(placeholderX=None):

    x = tflearn.input_data(shape=[None, IMG_SIZE_WIDTH, IMG_SIZE_HEIGHT, 1], name='input',
                           placeholder=placeholderX)

    x = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv1_1')
    x = tflearn.conv_2d(x, 64, 3, activation='relu', scope='conv1_2')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool1')

    x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_1')
    x = tflearn.conv_2d(x, 128, 3, activation='relu', scope='conv2_2')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool2')

    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_1')
    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_2')
    x = tflearn.conv_2d(x, 256, 3, activation='relu', scope='conv3_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool3')

    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_1')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_2')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv4_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool4')

    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_1')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_2')
    x = tflearn.conv_2d(x, 512, 3, activation='relu', scope='conv5_3')
    x = tflearn.max_pool_2d(x, 2, strides=2, name='maxpool5')

    x = tflearn.conv_2d(x, 4096, 7, activation='relu', scope='fc6')
    x = tflearn.dropout(x, 0.5, name='dropout1')

    x = tflearn.conv_2d(x, 4096, 1, activation='relu', scope='fc7')
    x = tflearn.dropout(x, 0.5, name='dropout2')

    x = tflearn.conv_2d(x, 2622, 1, activation='relu', scope='fc8')

    x = tflearn.flatten(x, name='Flatten')

    x = tflearn.activation(x, activation='softmax', name='Activation')

    return x


def classifier(img, faces_windows_location, scale=1.2, seuil_score=0.8, stepPixel=10):

    tf.reset_default_graph()

    convnet = own_model()

    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet, tensorboard_dir='log')

    if os.path.exists('{}.meta'.format(MODEL_LOCATION)):
        model.load(MODEL_LOCATION)
        print('model loaded!')

    # Size Conditions :
    height_max = img.shape[0]
    width_max = img.shape[1]

    # Defining windows list :
    img_windows = list()

    # Cropping parameters
    window_x = 36
    window_y = 36

    while window_x < width_max and window_y < height_max:

        # Crop image algorithm
        for x in range(0, width_max, stepPixel):  # pour chaque position vertical -> les carrees correspondants

            x_finish = x + window_x

            if x_finish > width_max:
                break

            for y in range(0, height_max, stepPixel):

                y_finish = y + window_y

                if y_finish <= height_max:

                    crop_img_original = img[y:y_finish, x:x_finish]

                    crop_img_transformed = cv2.resize(crop_img_original, (36, 36))

                    data = crop_img_transformed.reshape(IMG_SIZE_WIDTH, IMG_SIZE_HEIGHT, 1)

                    model_out = model.predict([data])[0]
                    face_ratio = model_out[0]

                    if face_ratio >= seuil_score:
                        img_windows.append([x, y, window_x, window_y, face_ratio])

                    print(x, y, window_x, window_y, face_ratio)

                else:
                    break

        # Edit Cropping parameters
        window_y = round(window_y * scale)
        window_x = round(window_x * scale)

    # Prepare data for clustering :
    with open(faces_windows_location, 'w') as f:
        f.write('x,y,width,height,score\n')

    with open(faces_windows_location, 'a') as f:
        for img_window in img_windows:
            f.write(
                '{},{},{},{},{}\n'.format(img_window[0], img_window[1], img_window[2], img_window[3], img_window[4]))



if __name__ == "__main__":

    print('main classifier here')
