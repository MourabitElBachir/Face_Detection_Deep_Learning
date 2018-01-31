from code.facedetector import *


def repo_face_detector(input_repo='data_detection', output_repo='images_results'):

    for img in os.listdir(input_repo):
        path = os.path.join(input_repo, img)
        print(path)
        face_detector_image_of_repo(path, output_repo)


if __name__ == '__main__':

    print(len(sys.argv))
    if (len(sys.argv)) == 1:
        repo_face_detector('../repo_test', '../output_results')

    elif (len(sys.argv)) == 3:
        repo_face_detector(sys.argv[1], sys.argv[2])

    else:
        print('error arguments specification !')