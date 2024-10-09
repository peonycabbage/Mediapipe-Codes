import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

from mediapipe.python.solutions.holistic import PoseLandmark
mp_pose = mp.solutions.pose
import numpy as np
import os






def extract_keypoints(results):
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    ls = np.array([results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].x, results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].y, results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_SHOULDER].z])
    rs = np.array([results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].x, results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].y, results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_SHOULDER].z])
    lw = np.array([results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].x, results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].y, results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_WRIST].z])
    rw = np.array([results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].x, results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].y, results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_WRIST].z])

    # Pose face landmarks
    face_landmarks_ids = [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_EYE_INNER,
        mp_pose.PoseLandmark.LEFT_EYE,
        mp_pose.PoseLandmark.LEFT_EYE_OUTER,
        mp_pose.PoseLandmark.RIGHT_EYE_INNER,
        mp_pose.PoseLandmark.RIGHT_EYE,
        mp_pose.PoseLandmark.RIGHT_EYE_OUTER,
        mp_pose.PoseLandmark.LEFT_EAR,
        mp_pose.PoseLandmark.RIGHT_EAR,
        mp_pose.PoseLandmark.MOUTH_LEFT,
        mp_pose.PoseLandmark.MOUTH_RIGHT
    ]
    
    face = np.array([[results.pose_landmarks.landmark[landmark].x, results.pose_landmarks.landmark[landmark].y, results.pose_landmarks.landmark[landmark].z] 
                     for landmark in face_landmarks_ids]).flatten() if results.pose_landmarks else np.zeros(len(face_landmarks_ids)*3)

    return np.concatenate([lh, rh, ls, rs, lw, rw, face])

# For static images:
IMAGE_FILES = []
BG_COLOR = (192, 192, 192) # gray
with mp_holistic.Holistic(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=True,
    refine_face_landmarks=True) as holistic:
  # for idx, file in enumerate(IMAGE_FILES):
  #   image = cv2.imread(file)
  #   image_height, image_width, _ = image.shape
  #   # Convert the BGR image to RGB before processing.
  #   results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    newcoordspath = r'C:\Research\Dataset\phoenix-2014-T.v3 (1).tar\PHOENIX-2014-T-release-v3\PHOENIX-2014-T\features\fullFrame-210x260px_KP'
    newpath = '/home/kartiaga/dataset/arabic190All_visualizeKP'
    path =r'C:\Research\Dataset\phoenix-2014-T.v3 (1).tar\PHOENIX-2014-T-release-v3\PHOENIX-2014-T\features\fullFrame-210x260px'
    for sets in os.listdir(path):
        setpath = os.path.join(path, sets)    
        for video in os.listdir(setpath):
            videopath = os.path.join(setpath, video)
            print("videopath", videopath) 
            vpath = os.listdir(videopath)
            for filename in os.listdir(videopath):
                #filenamepath = os.path.join(filename, videopath)
                #image = cv2.imread(filename)
                filenamepath = os.path.join(videopath, filename)
                image = cv2.imread(filenamepath)
                try:
                    image_height, image_width, _ = image.shape
                except AttributeError as e:
                    continue
                #print('filenamepath', filenamepath)
                results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                coordspath = os.path.join(newcoordspath, sets, video)
                if not os.path.exists(coordspath):
                    os.makedirs(coordspath)
                np.save(coordspath + "/" +str(filename), extract_keypoints(results))
                k = extract_keypoints(results)
                print('resultshape', k.shape)
                if results.pose_landmarks:
                    print('ok')

                annotated_image = image.copy()
    # Draw segmentation on the image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    #                 condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
    #                 bg_image = np.zeros(image.shape, dtype=np.uint8)
    #                 bg_image[:] = BG_COLOR
    #                 annotated_image = np.where(condition, annotated_image, bg_image)
    # # Draw pose, left and right hands, and face landmarks on the image.
    #                 # mp_drawing.draw_landmarks(
    #                 # annotated_image,
    #                 # results.face_landmarks,
    #                 # mp_holistic.FACEMESH_TESSELATION,
    #                 # landmark_drawing_spec=None,
    #                 # connection_drawing_spec=mp_drawing_styles
    #                 # .get_default_face_mesh_tesselation_style())
    #                 mp_drawing.draw_landmarks(
    #                     annotated_image,
    #                     results.pose_landmarks,
    #                     mp_holistic.POSE_CONNECTIONS,
    #                     landmark_drawing_spec=mp_drawing_styles.
    #                     get_default_pose_landmarks_style())
                    
    #                 newfilepath = os.path.join(newpath, sets, category, video)
    #                 if not os.path.exists(newfilepath):
                        
    #                     os.makedirs(newfilepath)
                    
    #                 print('newfiepath', newfilepath)
    #                 cv2.imwrite(newfilepath + '/'+ str(filename) + '.png', annotated_image)
    # # Plot pose world landmarks.
    #                 mp_drawing.plot_landmarks(
    #                     results.pose_world_landmarks, mp_holistic.POSE_CONNECTIONS)

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