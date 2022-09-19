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

# setting up the webcam to capture images
cap = cv2.VideoCapture(0)
# reading every frame
while True:
    success, img = cap.read()
    #recising the image to 0.25 of original
    imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    # find location of all the faces from our webcam feed [0] is not there so that
    # even if there are more than one faces in the feed
    facesCurrFrame = face_recognition.face_locations(imgS)
    # find encodings of the resized image sending in the small images and The location of
    # all the faces in current frame
    encodesCurrFrame = face_recognition.face_encodings(imgS, facesCurrFrame)




