import jetson.inference
import jetson.utils
import cv2
import numpy as np
import time

url='http://192.168.43.1:8080'      # put the ip link shown in the ipcamera app in the mobile after starting the server 
cam=cv2.VideoCapture(url+'/video')
net=jetson.inference.detectNet('ssd-mobilenet-v2',threshold=.5)
# cam.set(cv2.CAP_PROP_FRAME_WIDTH,dispW)
# cam.set(cv2.CAP_PROP_FRAME_HEIGHT,dispH)
font=cv2.FONT_HERSHEY_SIMPLEX
while True:
    _,frame=cam.read()
    frame=cv2.resize(frame,(480,240))
    height=frame.shape[0]
    width=frame.shape[1]
    img=cv2.cvtColor(frame,cv2.COLOR_BGR2RGBA).astype(np.float32)
    img=jetson.utils.cudaFromNumpy(img)
    detections=net.Detect(img,width,height)
    for detect in detections:
        print(detect)
        ID=detect.ClassID
        left=int(detect.Left)
        top=int(detect.Top)
        right=int(detect.Right)
        bottom=int(detect.Bottom)
        item=net.GetClassDesc(ID)
        tk=1
        cv2.rectangle(frame,(top,left),(bottom,right),(0,255,0),tk)
        cv2.putText(frame,item,(top,left+25),font,.75,(0,0,255),2)
    frame=cv2.resize(frame,(720,480))
    cv2.imshow('frame',frame)
    if cv2.waitKey(1)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()