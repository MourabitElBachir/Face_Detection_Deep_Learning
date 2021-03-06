# Face Detection Deep Learning
Implementation of a face detector based on a convolutional neural network

## Project Members

+ MOURABIT EL BACHIR
+ DAIF ARIJ
+ BELGHARIB ZAKARIA

## Aim

This project offers a deep system to detect faces from images using a base of reduced size images. We have been able to obtain encouraging results despite some small difficulties due to the limitations of our methods but which are leading to future improvements.

Since this is also a class project, we wanted to learn the several things from it :

- Learn hard functionalities of Python; 
- Training: Implement many models of neural networks to obtain good results; 
- Crop and Scale algorithms to classify faces in images.
- Algorithms of clustering such as DBSCAN and MeanShift.
- OpenCV functionnalities for showing images and create rectangles when finding faces.

## Running the project:

### Prerequisites

	- Python 3.6
	- Tensorflow/TFlearn
	- Ideally, a GNU/Linux based operating system, though the code is meant to be portable
  

### Execution

##### Simple Image :

- Console mode:
	+ access the project repository in the console  
	+ run "python facedetector.py {image-src-location} {repository-to-store-image}" with the python interpreter in you installed 		everything.

- PyCharm mode:
  + run "facedetector.py" with the main program.
  
##### Repository of images :
 
 - Console mode:
	+ access the project repository in the console  
	+ run "python repositorydetector.py {repository-src-location} {repository-to-store-images}" with the python interpreter in you 		installed everything.

 - PyCharm mode:
  + run "repositorydetector.py" with the main program.
  
## Description of the packages

+ #### Code: 
	
   Contains all files to execute the project and also contain 'log/' repository to use with tensorBoard if we want to show graphs 	    of accuracy and loss

  - Le fichier "env_variables.py" : Define all important variable to start the main functionalities of our deep face detector

  - Le fichier "facedetector.py" : The main file to detect faces in a image
  
  - Le fichier "classify.py" : To classify the windows matching faces with a score > based_score
  
  - Le fichier "clustering.py" : To Clean the detection when lot of windows detecting the same face
  
  - Le fichier "display.py" : To display the rectangles matching faces in the images and also storing images results and text traces

  - Le fichier "repositorydetector.py" : The main file to detect faces in all images of a repository 

  - Le fichier "preparing_data.py" : Storing the image in a file format to use the database in training part
  
  - Le fichier "training.py" : The training part, at the last step we save the model to use it in detection tests part

+ #### models :

	- Contains all neural network models, and we use the model specified in the "env_variables.py" file

+ #### Data :

  - Contains two repository : Train, Test

  - The train repository : Should contain two repository ['face', 'notFace']
  
  - The test repository : Should contain two repository ['face', 'notFace'] 
  
+ #### Informations :

  - Contains the training and test database in a '.npy' file format. 


## Running the project:

### Result 1 : 
 
![Result1](https://github.com/MourabitElBachir/Face_Detection_Deep_Learning/blob/master/images_output/President_Obama_result.png)

### Result 2 :

![Result2](https://github.com/MourabitElBachir/Face_Detection_Deep_Learning/blob/master/images_output/Friends_season_one_cast_result.jpg)

### Result 3 :

![Result3](https://github.com/MourabitElBachir/Face_Detection_Deep_Learning/blob/master/images_output/Obama_result.jpg)
  
  
## Evaluation graphs for training and testing :


### Evaluation 1 : 

![Result3](https://github.com/MourabitElBachir/Face_Detection_Deep_Learning/blob/master/evaluation/evaluation1.PNG)

### Evaluation 2 : 
  
![Result3](https://github.com/MourabitElBachir/Face_Detection_Deep_Learning/blob/master/evaluation/evaluation2.PNG)

### Evaluation 3 : 
  
![Result3](https://github.com/MourabitElBachir/Face_Detection_Deep_Learning/blob/master/evaluation/evaluation3.PNG)
  
