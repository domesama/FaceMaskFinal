#Import libraries
import cv2
import os
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
 #Define local directory paths
faceCascade = cv2.CascadeClassifier("C:/Users/THETHAWATKONGYU/Desktop/GitHub Repositories/FaceMaskFinal/haarcascade_frontalface_default.xml")
model = load_model("C:/Users/THETHAWATKONGYU/Desktop/GitHub Repositories/FaceMaskFinal/mask_recog_ver1.h5")
 
#Asks OpenCV to look for a webcam at position (0) if there's more we can use the index (1)
video_capture = cv2.VideoCapture(0)
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Cuts out frames which includes face usinging detectMultiScale() with the Harrcascade loaded from our directory mentioned above.
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    listOfFace=[]
    preds=[]

    #Loop over the face for each dimension of detected face in the given detected frame and append it into listOfFace array
    for (x, y, w, h) in faces:
        knownFace = frame[y:y+h,x:x+w]
        knownFace = cv2.cvtColor(knownFace, cv2.COLOR_BGR2RGB)
        knownFace = cv2.resize(knownFace, (224, 224))
        knownFace = img_to_array(knownFace)
        knownFace = np.expand_dims(knownFace, axis=0)
        knownFace =  preprocess_input(knownFace)
        listOfFace.append(knownFace)

    #If there's a face we model.predict() the cut out picture (Preproccessed from the above) with our Tensorflow trained model.
        if len(listOfFace)>0:
            preds = model.predict(listOfFace)

    #For each predictions that we make preds() we procceed to draw out the frames and mark the color according to the predicted class.
        for pred in preds:
            (mask, withoutMask) = pred
        label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
        cv2.putText(frame, label, (x, y- 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
 
        cv2.rectangle(frame, (x, y), (x + w, y + h),color, 2)
    # Display the resulting frame after all the compuation is done on the GUI
    cv2.imshow('Video', frame)
