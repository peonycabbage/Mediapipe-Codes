# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 10:03:25 2023

@author: Keren
"""

import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

from mediapipe.python.solutions.holistic import PoseLandmark
import numpy as np
import os


def mediapipe_detection(image, model):
    print("In mediapipe_detection function")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable (just to save on some memory)
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results



def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    ls = np.array([results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x, results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y, results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].z])
    rs = np.array([results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x, results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y, results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].z])
    lw = np.array([results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].x, results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].y, results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].z])
    rw = np.array([results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].x, results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].y, results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].z])
   
    
    return np.concatenate([ lh, rh, ls, rs,lw, rw ])


# For webcam input:
#cap = cv2.VideoCapture(0)



newcoordspath = r"D:\PhD Applications\Munster Uni\References\CSL8KP"
newpath = r"D:\PhD Applications\Munster Uni\References\CSL8_visualizeKP"
path =r"D:\PhD Applications\Munster Uni\References\CSL Dataset for iconicity"

for sets in os.listdir(path):
    setpath = os.path.join(path, sets)
    print('setpath', setpath)
    for category in os.listdir(setpath):
       categorypath = os.path.join(setpath, category)
       print('categorypath', categorypath)
       #print('categpath', categorypath)
       count = 0
       for video in os.listdir(categorypath):
            videopath = os.path.join(categorypath, video)
            #print("videopath", videopath) 
            #vpath = os.listdir(videopath)
            cap = cv2.VideoCapture(videopath)
            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print("LENGTH:" + str(length))
            
            
            with mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5) as holistic:
            
            
               
                while cap.isOpened():
                    count=count+1
                    print("Reading feed")          
                    ret, frame = cap.read()
                    if(frame is None):
                        print('no image')
                        break
                    image, results = mediapipe_detection(frame, holistic)
                    #extract_keypoints(results)
                    
                    
                    coordspath = os.path.join(newcoordspath, sets, category, video)
                    if not os.path.exists(coordspath):
                        os.makedirs(coordspath)
                    
                    if results.pose_landmarks:
                        np.save(coordspath + "/" +str(video) + str(count), extract_keypoints(results))
                    
                       
                        annotated_image = image.copy()
        # Draw segmentation on the image.
        # To improve segmentation around boundaries, consider applying a joint
        # bilateral filter to "results.segmentation_mask" with "image".
                        condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
                        #bg_image = np.zeros(image.shape, dtype=np.uint8)
                        #bg_image[:] = BG_COLOR
                        #annotated_image = np.where(condition, annotated_image, bg_image)
        # Draw pose, left and right hands, and face landmarks on the image.
                        # mp_drawing.draw_landmarks(
                        # annotated_image,
                        # results.face_landmarks,
                        # mp_holistic.FACEMESH_TESSELATION,
                        # landmark_drawing_spec=None,
                        # connection_drawing_spec=mp_drawing_styles
                        # .get_default_face_mesh_tesselation_style())
                        mp_drawing.draw_landmarks(
                            annotated_image,
                            results.pose_landmarks,
                            mp_holistic.POSE_CONNECTIONS,
                            landmark_drawing_spec=mp_drawing_styles.
                            get_default_pose_landmarks_style())
                        
                        newfilepath = os.path.join(newpath, sets, category, video)
                        if not os.path.exists(newfilepath):
                            
                            os.makedirs(newfilepath)
                        
                        print('newfiepath', newfilepath)
                        cv2.imwrite(newfilepath + '/'+ str(video) + str(count) + '.png', annotated_image)
        # Plot pose world landmarks.
                        mp_drawing.plot_landmarks(
                            results.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS)
                    else:
                       break





#cap.release()

# For webcam input:
# cap = cv2.VideoCapture(0)
# with mp_holistic.Holistic(
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5) as holistic:
#   while cap.isOpened():
#     success, image = cap.read()
#     if not success:
#       print("Ignoring empty camera frame.")
#       # If loading a video, use 'break' instead of 'continue'.
#       continue

#     # To improve performance, optionally mark the image as not writeable to
#     # pass by reference.
#     image.flags.writeable = False
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = holistic.process(image)

#     # Draw landmark annotation on the image.
#     image.flags.writeable = True
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     mp_drawing.draw_landmarks(
#         image,
#         results.face_landmarks,
#         mp_holistic.FACEMESH_CONTOURS,
#         landmark_drawing_spec=None,
#         connection_drawing_spec=mp_drawing_styles
#         .get_default_face_mesh_contours_style())
#     mp_drawing.draw_landmarks(
#         image,
#         results.pose_landmarks,
#         mp_holistic.POSE_CONNECTIONS,
#         landmark_drawing_spec=mp_drawing_styles
#         .get_default_pose_landmarks_style())
#     # Flip the image horizontally for a selfie-view display.
#     cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
#     if cv2.waitKey(5) & 0xFF == 27:
#       break
# cap.release()