import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

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
    # noinspection PyUnresolvedReferences
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

def markAttendance(name):
    with open('Att.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')



encodeListKnown = enc(images)
print("Done with encoding")

# setting up the webcam to capture images
cap = cv2.VideoCapture(0)
# reading every frame
while True:
    success, img = cap.read()
    # resizing the image to 0.25 of original
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    # find location of all the faces from our webcam feed [0] is not there so that
    # even if there are more than one faces in the feed
    facesCurrFrame = face_recognition.face_locations(imgS)
    # find encodings of the resized image sending in the small images and The location of
    # all the faces in current frame
    encodesCurrFrame = face_recognition.face_encodings(imgS, facesCurrFrame)
    # one by one it grabs faces from facesCurrFrame list and Encodings From encodesCurrFrame list
    for enc, faceLoc in zip(encodesCurrFrame, facesCurrFrame):
        matches = face_recognition.compare_faces(encodeListKnown, enc)
        # this will return a list which will contain the distance or dis-similaritis the image from camera
        # have and the images provided in the training data
        # the lowest value will have our match
        faceDis = face_recognition.face_distance(encodeListKnown, enc)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
           #print(name)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
