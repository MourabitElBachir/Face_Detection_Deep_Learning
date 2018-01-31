# Traning data directory
TRAIN_DIR = '../data/train'

# Validation data for neural network
TEST_DIR = '../data/test'

# Database for train and test :
TRAIN_DB = '../informations/train_data.npy'
TEST_DB = '../informations/test_data.npy'

# Our topic classes
CLASSES = ['face', 'notFace']

# the same fixed size of all trainig and test image
IMG_SIZE_WIDTH = 36
IMG_SIZE_HEIGHT = 36

#Defining the learning rate
LR = 1e-5

# Just so we remember which saved model is which, sizes must match
MODEL_NAME = 'faceDetection-{}-{}.model'.format(LR, '2conv-basic')
MODEL_LOCATION = '../models/{}'.format(MODEL_NAME)

# RECTANGLES
RECTANGLES = ['x', 'y', 'width', 'height', 'score']

# Clustering :

#DEFAULT EPS for clustering distances
DEFAULT_DBSCAN_EPS = 0.5
# 0 For mean cluster representative method and 1 for argmax
CLUSTER_REPRESENTATIVE_METHOD = 0

# 1: DBSCAN, 0: MeanShift
DBSCAN_OR_MeanShift = 1

