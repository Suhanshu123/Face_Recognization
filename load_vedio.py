# -*- coding: utf-8 -*-
"""
Created on Thu May 28 11:40:51 2020

@author: Suhanshu
"""
import numpy as np
import cv2
import os

import face_recognization as fr
print (fr)



face_recognizer=cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read('C:/Users/Suhanshu/Desktop/trainig.yml')    #Give path of where trainingData.yml is saved

cap=cv2.VideoCapture(0)   #If you want to recognise face from a video then replace 0 with video path

name={0:"Suhanshu",1:"ben_afflec",2:"elton_john",3:"jerry_seinfield",4:"mandonna",5:"mindy_kaling"}             #Change names accordingly.  If you want to recognize only one person then write:- name={0:"name"} thats all. Dont write for id number 1. 
#Change names accordingly.  If you want to recognize only one person then write:- name={0:"name"} thats all. Dont write for id number 1.
while True:
    ret,test_img=cap.read()
    faces_detected,gray_img=fr.faceDetection(test_img)
    print("face Detected: ",faces_detected)
    for (x,y,w,h) in faces_detected:
        cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,255,0),thickness=5)

    
    
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
    if cv2.waitKey(10)==ord('q'):
        break