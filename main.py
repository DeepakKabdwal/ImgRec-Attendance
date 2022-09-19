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
# print(classNames)
# start our encoding process

def enc(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


encodeListKnown = enc(images)
print("Done with encoding")





