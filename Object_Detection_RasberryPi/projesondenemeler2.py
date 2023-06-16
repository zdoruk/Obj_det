import cv2
import json

import RPi.GPIO as GPIO
from datetime import datetime
from picamera import PiCamera
from time import sleep
import os
import pyrebase
import json

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BOARD)
GPIO.setup(10, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)

firebaseConfig = {
        'apiKey': "AIzaSyD-4ISghw9sUUhqoJP1LDnF00s9Jcf8Tr4",
        'authDomain': "smart-refrigerator-app-db.firebaseapp.com",
        'databaseURL': "https://smart-refrigerator-app-db.firebaseio.com",
        'projectId': "smart-refrigerator-app-db",
        'storageBucket': "smart-refrigerator-app-db.appspot.com",
        'messagingSenderId': "792819827695",
         'appId': "1:792819827695:web:ddf3b58328dbb34f302c1a",
        'measurementId': "G-MH88TSZCLS"
        }

firebase = pyrebase.initialize_app(firebaseConfig)
storage = firebase.storage()

#thres = 0.45 # Threshold to detect object

classNames = []
classFile = "/home/zeynel/Desktop/Object_Detection_Files/coco.names"
with open(classFile,"rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

configPath = "/home/zeynel/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/zeynel/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def getObjects(img, thres, nms, draw=True, objects=[]):
    classIds, confs, bbox = net.detect(img,confThreshold=thres,nmsThreshold=nms)
    #print(classIds,bbox)
    if len(objects) == 0: objects = classNames
    objectInfo =[]
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box,className])
                if (draw):
                    cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                    cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                    cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    return img,objectInfo


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
    #cap.set(10,70)

    x=0
    while True:
        success, img = cap.read()
        result, objectInfo = getObjects(img,0.45,0.2)
        #print(objectInfo)
        
        #print(objectInfo)
        print("Tespit Edilen Nesneler:", objectInfo)
        cv2.imshow("Output", img)
        cv2.imshow("Output",img)
        cv2.waitKey(1)
        x=x+1
        if x==5:
            # Nesne bilgilerini JSON dosyasına kaydetme
            icerik_listesi=[]
            for item in objectInfo:
                icerik=item[1]
                icerik_listesi.append(icerik)
            
            now = datetime.now()
            dtd = now.strftime("%d")
            dta = now.strftime("%m")
            dty = now.strftime("%Y")
            dth = now.strftime("%H")
            dtm = now.strftime("%M")
            dts = now.strftime("%S")
            
            image_name = "HiFvucuVzVU4XBnuzyENc8IXOXq2-r.jpg"
            json_name = "HiFvucuVzVU4XBnuzyENc8IXOXq2-j.json"
            
            json_data = {
                "food": icerik_listesi,
                "date": {
                            "year":dty,
                            "month":dta,
                            "day":dtd,
                            "hour":dth,
                            "minute":dtm,
                            "second":dts
                            }
                        }
            with open(json_name, "w") as json_file:
                json.dump(json_data, json_file)
        
            storage.child(json_name).put(json_name)
            print("JSON file sent")
            
            
            
            
            cv2.imwrite("HiFvucuVzVU4XBnuzyENc8IXOXq2-r.jpg",img)
            print(image_name + " saved")
            storage.child(image_name).put(image_name)
            print("Image sent")
            break
    cap.release()
    cv2.destroyAllWindows()
    
if 1==1:
    
    camera = PiCamera()
    image_name2 = "HiFvucuVzVU4XBnuzyENc8IXOXq2-p.jpg"
    camera.capture(image_name2)
    print(image_name + " saved")
    storage.child(image_name2).put(image_name2)
    print("Image2 sent")
    






