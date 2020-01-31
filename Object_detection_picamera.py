######## Picamera Object Detection Using Tensorflow Classifier #########
#
# Author: Evan Juras
# Date: 4/15/18
# Description: 
# This program uses a TensorFlow classifier to perform object detection.
# It loads the classifier uses it to perform object detection on a Picamera feed.
# It draws boxes and scores around the objects of interest in each frame from
# the Picamera. It also can be used with a webcam by adding "--usbcam"
# when executing this script from the terminal.

## Some of the code is copied from Google's example at
## https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

## and some is copied from Dat Tran's example at
## https://github.com/datitran/object_detector_app/blob/master/object_detection_app.py

## but I changed it to make it more understandable to me.


# Import packages
import os
import cv2 
import numpy as np
from picamera.array import PiRGBArray
from picamera import PiCamera
import tensorflow as tf
import argparse
import pytesseract
import sys
import time
import subprocess
from PIL import Image

# Set up camera constants
IM_WIDTH = 352
IM_HEIGHT = 480
#IM_WIDTH = 640    Use smaller resolution for
#IM_HEIGHT = 480   slightly faster framerate

# Select camera type (if user enters --usbcam when calling this script,
# a USB webcam will be used)
camera_type = 'picamera'
parser = argparse.ArgumentParser()
parser.add_argument('--usbcam', help='Use a USB webcam instead of picamera',
                    action='store_true')
args = parser.parse_args()
if args.usbcam:
    camera_type = 'usb'

# This is needed since the working directory is the object_detection folder.
sys.path.append('..')

# Import utilites
from utils import label_map_util
from utils import visualization_utils as vis_util

# Name of the directory containing the object detection module we're using
MODEL_NAME = 'plate'

# Grab path to current working directory
CWD_PATH = os.getcwd()

# Path to frozen detection graph .pb file, which contains the model that is used
# for object detection.
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,'frozen_inference_graph.pb')

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,'training','labelmap.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 1

## Load the label map.
# Label maps map indices to category names, so that when the convolution
# network predicts `5`, we know that this corresponds to `airplane`.
# Here we use internal utility functions, but anything that returns a
# dictionary mapping integers to appropriate string labels would be fine
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)


# Define input and output tensors (i.e. data) for the object detection classifier

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

# Output tensors are the detection boxes, scores, and classes
# Each box represents a part of the image where a particular object was detected
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

# Each score represents level of confidence for each of the objects.
# The score is shown on the result image, together with the class label.
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

# Number of objects detected
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize camera and perform object detection.
# The camera has to be set up and used differently depending on if it's a
# Picamera or USB webcam.

# I know this is ugly, but I basically copy+pasted the code for the object
# detection loop twice, and made one work for Picamera and the other work
# for USB.

### Picamera ###
if camera_type == 'picamera':
    # Initialize Picamera and grab reference to the raw capture
    camera = PiCamera()
    camera.resolution = (IM_WIDTH,IM_HEIGHT)
    camera.framerate = 10
    rawCapture = PiRGBArray(camera, size=(IM_WIDTH,IM_HEIGHT))
    rawCapture.truncate(0)

    for frame1 in camera.capture_continuous(rawCapture, format="jpeg",use_video_port=True):

        t1 = cv2.getTickCount()
        
        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        frame = np.copy(frame1.array)
        frame.setflags(write=1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_expanded = np.expand_dims(frame_rgb, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.40)

        cv2.putText(frame,"FPS: {0:.2f}".format(frame_rate_calc),(10,325),font,1,(255,255,0),2,cv2.LINE_AA)

        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('Object detector', frame)

        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

        rawCapture.truncate(0)

    camera.close()

### USB webcam ###
elif camera_type == 'usb':
    # Initialize USB webcam feed
    video = cv2.VideoCapture(0)
    ret = video.set(3,IM_WIDTH)
    ret = video.set(4,IM_HEIGHT)

    while(True):

        t1 = cv2.getTickCount()

        # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
        # i.e. a single-column array, where each item in the column has the pixel RGB value
        ret, frame = video.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_expanded = np.expand_dims(frame, axis=0)

        # Perform the actual detection by running the model with the image as input
        (boxes, scores, classes, num) = sess.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})
        
        
        (frame_height, frame_width) = frame.shape[:2]
        for i in range(len(np.squeeze(scores))):

            #print(np.squeeze(boxes)[i])
            ymin = int((np.squeeze(boxes)[0][0]*frame_height))
            xmin = int((np.squeeze(boxes)[0][1]*frame_width))
            ymax = int((np.squeeze(boxes)[0][2]*frame_height))
            xmax = int((np.squeeze(boxes)[0][3]*frame_width))
            cropped_img = frame[ymin:ymax,xmin:xmax]
             
        #cropped_image = "{}.png".format(os.getpid())
        cv2.imwrite('cropped_image.jpg', cropped_img)
       

                    
        #cv2.imshow('cropped image.jpg', cropped_img)  
    
        
        #mg=cv2.imread('cropped_img.png')   
        
        #Read the number plate
        
        grayimg = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        
        thresh = cv2.threshold(grayimg, 10, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        
        thresh1 = cv2.GaussianBlur(thresh,(5,5),0)
        
        
        kernel = np.ones((2,2),np.uint8)
        erosian= cv2.erode(thresh1,kernel,iterations = 1)

        
        
               
        configr = (' lang=eng oem 0 --psm 6-c tessedit_char_whitelist=BZY1234567890')
        #b = ['B', 'A',]
        #textlist = []
      
        text = pytesseract.image_to_string(erosian, config='configr')
        print("Detected Number is:",text)
        if text=='B-6014 NZY':
                print('Nama Pemilik kendaraan: Cecep')
                print('Jenis kendaraan: Vespa')
                #break
        #subprocess.call(["curl","-X" POST "https://api.thebigbox.id/sms-broadcast/1.0.0/send" -H "accept: application/json" -H "x-api-key: t2tNxEtL3S35h0Z83vy57pF9MiwDD3Yq" -H "Content-Type: application/json" -d "{ \"smsblast_username\": \"\", \"smsblast_password\": \"\", \"smsblast_senderid\": \"\", \"msisdns\": [ \"085695848790\" ], \"text\": \"melanggar\"}"]))])
                subprocess.call(["curl", "-X", "POST", "https://api.thebigbox.id/sms-broadcast/1.0.0/send", "-H", "accept: application/json", "-H", "x-api-key: t2tNxEtL3S35h0Z83vy57pF9MiwDD3Yq", "-H", "Content-Type: application/json", "-d", "{ \"smsblast_username\": \"\", \"smsblast_password\": \"\", \"smsblast_senderid\": \"\", \"msisdns\": [ \"085695848790\" ], \"text\": \"melanggar\"}"])
                break
   #print("sukses")


        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    


            #rrbreak

        # Draw the results of the detection (aka 'visulaize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.85)
        
        #print(type(text))
        #textlist.append(text)
        #print("Detected Number is:",textlist)
        #a=("B-6014 NZY")
        #if text==type(str) and text[0] == 'b':
        #    print('jakarta nich')
            #rrbreak
        
        
     
        # All the results have been drawn on the frame, so it's time to display it.
        cv2.imshow('cropped image.jpg', cropped_img)
        cv2.imshow('Object detector', frame)
        
        t2 = cv2.getTickCount()
        time1 = (t2-t1)/freq
        frame_rate_calc = 1/time1
         
        #subprocess.call([curl -X POST "https://api.thebigbox.id/sms-broadcast/1.0.0/send" -H "accept: application/json" -H "x-api-key: t2tNxEtL3S35h0Z83vy57pF9MiwDD3Yq" -H "Content-Type: application/json" -d "{ "smsblast_username": "cecep", "smsblast_password": "b-2929", "smsblast_senderid": "anda","msisdns":"085695848790" , "text": "anda melanggar"}"])

        # Press 'q' to quit
        if cv2.waitKey(1) == ord('q'):
            break

    

cv2.destroyAllWindows()


