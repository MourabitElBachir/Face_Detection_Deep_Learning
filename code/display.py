from code.env_variables import *
import pandas
import cv2


def display_recognition(input_image, output_image, faces_windows_path, show_image=1):

    img = cv2.imread(input_image)

    data = pandas.read_csv(filepath_or_buffer=faces_windows_path, delimiter=',', encoding='utf-8')
    print(data)

    windows = data[RECTANGLES].values
    print(windows)
    windows = windows.astype(int)

    i = 1

    input_image = input_image[input_image.rfind("\\") + 1:]
    filename = input_image[input_image.rfind("/") + 1:]
    filename_small = filename.replace(".", "_result.")

    image_trace = filename_small.split('.')[0]
    with open(output_image+'/'+image_trace+'.txt', 'w') as f:
        f.write('Image : '+image_trace+'\n')

    for window in windows:

        print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, window[0], window[1],
                                                                                 window[0] + window[2], window[1] + window[3]))

        with open(output_image + '/' + image_trace + '.txt', 'a') as f:
            f.write("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, window[0], window[1],
                                                                                 window[0] + window[2], window[1] + window[3]) + '\n')

        cv2.rectangle(img, (window[0], window[1]), (window[0] + window[2], window[1] + window[3]), (255, 0, 0), 2)

        i += 1

    cv2.imwrite(output_image+'/'+filename_small, img)

    if show_image == 1:
        cv2.imshow("Display", img)
        cv2.waitKey(0)


if __name__ == '__main__':

	display_recognition('../single_test/Friends_season_one_cast.jpg', '../images_results', '../informations/faces_windows_clustering_result.csv')