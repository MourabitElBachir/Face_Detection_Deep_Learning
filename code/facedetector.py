from code.classifier import *
from code.clustering import *
from code.display import *
import sys
import csv


def face_detector_image(input_image='data_detection/image.jpg', output_image='images_results'):

    # Classifing image :
    faces_windows_input = '../informations/match_faces_windows.csv'
    img = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    classifier(img, faces_windows_input)

    # Clustering image :
    faces_windows_output = '../informations/faces_windows_clustering_result.csv'
    clustering(DBSCAN_OR_MeanShift, faces_windows_input, faces_windows_output)

    # Display Result :
    display_recognition(input_image, output_image, faces_windows_output, 1)


def face_detector_image_of_repo(input_image='data_detection', output_image='images_results'):

    # Classifing image :
    faces_windows_input = '../informations/match_faces_windows.csv'
    img = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    classifier(img, faces_windows_input)

    # Clustering image :
    faces_windows_output = '../informations/faces_windows_clustering_result.csv'
    clustering(DBSCAN_OR_MeanShift, faces_windows_input, faces_windows_output)

    # Display Result :
    display_recognition(input_image, output_image, faces_windows_output, 0)

if __name__ == '__main__':

    print(len(sys.argv))
    if (len(sys.argv)) == 1:
        face_detector_image('../single_test/28-10-16-499784761.jpg', '../images_results')

    elif (len(sys.argv)) == 3:
        face_detector_image(sys.argv[1], sys.argv[2])

    else:
        print('error arguments specification !')


