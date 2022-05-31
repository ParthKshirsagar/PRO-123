from cv2 import COLOR_BGR2GRAY
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps 
import os, ssl, time
import cv2

#Reading the data
X = np.load('image.npz')['arr_0']
y = pd.read_csv('labels.csv')['labels']
print(pd.Series(y).value_counts())
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
nclasses = len(classes)

#Splitting the data into test and train
X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=9, train_size=7500, test_size=2500)

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

#Fitting X_train and Y_train into Logistic Regression model
clf = LogisticRegression(solver='saga', multi_class='multinomial').fit(X_train_scaled, Y_train)

#Checking the accuracy score of the Logistic Regression model
y_pred = clf.predict(X_test_scaled)
accuracy = accuracy_score(Y_test, y_pred)
print(f'Accuracy: {accuracy}')

#Starting the camera

cap = cv2.VideoCapture(0)

while(True):
    try:
        ret, frame = cap.read()

        #Operations on the frame
        gray = cv2.cvtColor(frame, COLOR_BGR2GRAY)

        #Drawing a rectangular box in the middle
        height, width = gray.shape
        upper_left = (int(width/2-56), int(height/2-56))
        bottom_right = (int(width/2+56), int(height/2+56))
        cv2.rectangle(gray, upper_left, bottom_right, (0, 255, 0), 2)

        #Considering the area indside the box
        roi = gray[upper_left[1]:bottom_right[1], upper_left[0]:bottom_right[0]]

        #Converting cv2 to PIL format
        img_pil = Image.fromarray(roi)

        #Converting the video capture to grayscale. 'L' format means each pixel is represented by a single value from 0 to 255
        img_bw = img_pil.convert('L')
        img_bw_resized = img_bw.resize((28, 28), Image.ANTIALIAS)
        #Invert the image
        img_bw_resized_inverted = PIL.ImageOps.invert(img_bw_resized)
        pixel_filter = 20
        #converting to scalar quantity
        min_pixel = np.percentile(img_bw_resized_inverted, pixel_filter)
        #using clip to limit the values between 0,255
        img_bw_resized_inverted_scaled = np.clip(img_bw_resized_inverted-min_pixel, 0, 255)
        max_pixel = np.max(img_bw_resized_inverted)
        #converting into an array
        img_bw_resized_inverted_scaled = np.asarray(img_bw_resized_inverted_scaled)/max_pixel
        #creating a test sample and making a prediction
        test_sample = np.array(img_bw_resized_inverted_scaled).reshape(1, 784)
        test_pred = clf.predict(test_sample)
        print(f'Predicted class is: {test_pred}')

        cv2.imshow('frame', gray)
        if(cv2.waitKey(1) and 0xFF == ord('q')):
            break
    except Exception as e:
        pass

cap.release()
cv2.destroyAllWindows()