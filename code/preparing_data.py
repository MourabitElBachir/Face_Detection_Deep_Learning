import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
from code.env_variables import *

def label_img(cls):

    # conversion to array [face, notFace]
    #                            [much face, no notFace]
    if cls == 'face': return [1,0]
    #                             [no cat, very doggo]
    elif cls == 'notFace': return [0,1]


def create_train_data():
    training_data = []

    for cls in CLASSES:
        for img in tqdm(os.listdir("{}/{}".format(TRAIN_DIR, cls))):
            label = label_img(cls)
            path = os.path.join("{}/{}".format(TRAIN_DIR, cls), img)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE_WIDTH, IMG_SIZE_HEIGHT))
            training_data.append([np.array(img), np.array(label)])

    shuffle(training_data)
    np.save(TRAIN_DB, training_data)
    return training_data


def process_test_data():
    testing_data = []

    for cls in CLASSES:
        for img in tqdm(os.listdir("{}/{}".format(TEST_DIR, cls))):
            label = label_img(cls)
            path = os.path.join("{}/{}".format(TEST_DIR, cls), img)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE_WIDTH, IMG_SIZE_HEIGHT))
            testing_data.append([np.array(img), np.array(label)])

    shuffle(testing_data)
    np.save(TEST_DB, testing_data)
    return testing_data


if __name__ == "__main__":
    train_data = create_train_data()
    test_data = process_test_data()