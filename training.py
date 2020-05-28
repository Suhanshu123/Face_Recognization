# -*- coding: utf-8 -*-
"""
Created on Wed May 27 21:14:43 2020

@author: Suhanshu
"""

import numpy as np
import cv2
import os

import face_recognization as fr

test_img=cv2.imread("C:/Users/Suhanshu/Desktop/intro.jpg")

faces_detected,gray_img=fr.faceDetection(test_img)

#Training will begin from here

faces,faceID=fr.labels_for_training_data("C:/Users/Suhanshu/Desktop/Images") #Give path to the train-images folder which has both labeled folder as 0 and 1
face_recognizer=fr.train_classifier(faces,faceID)
face_recognizer.save('C:/Users/Suhanshu/Desktop/trainig.yml') #It will save the trained model. Just give path to where you want to save

name={0:"Suhanshu",1:"ben_afflec",2:"elton_john",3:"jerry_seinfield",4:"mandonna",5:"mindy_kaling"}    #Change names accordingly. If you want to recognize only one person then write:- name={0:"name"} thats all. Dont write for id number 1.


for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)
    print ("Confidence :",confidence)
    print("label :",label)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]
    fr.put_text(test_img,predicted_name,x,y)

resized_img=cv2.resize(test_img,(1000,700))
cv2.imshow("face detection ", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows