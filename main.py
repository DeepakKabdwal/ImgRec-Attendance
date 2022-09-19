import cv2
import face_recognition
import numpy as np
import os

path = 'imagesAttendance'
# list to store images
images = []
# list to store names
classNames = []
# path to the directory of training data
myList = os.listdir(path)
# print(myList)
# importing every image in the training directory
for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    # basically removing the extension from the image it luks gud
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

