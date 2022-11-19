# pyth
facial recognition using open cv
***////i swear i'll add something here later but for now just take my word it knows faces.////***

To use first install and set up any python interpretor 
then download OpenCV, dlib, face-recognition, numpy
use "pip install -----" followed by the name of the library

then get some training images that you want the program to compare from the webcam feed and name them properly as it gives out the name of the image as Name.
The images folder should be on the same directory where the code is located otherwise give the path to the directory containing training images in "path" variable.


Run the program once it generates encodings for training data it will start showing results on screen or in an CSV(comma seperated values) if you like. Or I'll branch out an update for that too.

Only the names are stored in the Att.csv file now.

if you want to see the feed on the screen 

\*\  # cv2.imshow('Webcam', img)
    # cv2.waitKey(1)
/*/

uncomment these two lines.

Softwares Used:-
1. Pycharm Community Edition
2. Python 3.9
3. Anaconda (for environment).

References:-
https://medium.com/@ageitgey/machine-learning-is-fun-80ea3ec3c471

{Machine Learning is fun article on medium}

[![@dezcvr's Holopin board](https://holopin.me/dezcvr)](https://holopin.io/@dezcvr)
