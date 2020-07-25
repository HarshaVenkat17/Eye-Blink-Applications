#please open dino game before running this code

import cv2
import numpy as np
import dlib
import time
import pyautogui
from math import hypot

capture = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midpoint(p1 ,p2):
    return int((p1.x + p2.x)/2), int((p1.y + p2.y)/2)

def blink_ratio(eye_points, facial_landmarks):
    left_point = (facial_landmarks.part(eye_points[0]).x, facial_landmarks.part(eye_points[0]).y)
    right_point = (facial_landmarks.part(eye_points[3]).x, facial_landmarks.part(eye_points[3]).y)
    center_top = midpoint(facial_landmarks.part(eye_points[1]), facial_landmarks.part(eye_points[2]))
    center_bottom = midpoint(facial_landmarks.part(eye_points[5]), facial_landmarks.part(eye_points[4]))
    horiontal= cv2.line(img, left_point, right_point, (0,255, 0), 2)
    vertical = cv2.line(img, center_top, center_bottom, (0,255, 0), 2)

    hlength = hypot((left_point[0] - right_point[0]), (left_point[1] - right_point[1]))
    vlength = hypot((center_top[0] - center_bottom[0]), (center_top[1] - center_bottom[1]))
    return (hlength/vlength)

while True:
    img = capture.read()[1]
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(imgray)
    for i in faces:
        landmarks = predictor(imgray,i)
        left_eye_ratio = blink_ratio([36, 37, 38, 39, 40, 41], landmarks)
        right_eye_ratio = blink_ratio([42, 43, 44, 45, 46, 47], landmarks)
        if left_eye_ratio>4:
            pyautogui.keyDown("up")
        elif right_eye_ratio>4:
            pyautogui.keyDown("down")
            time.sleep(0.5)
            pyautogui.keyUp("down")
    #cv2.imshow("Image",img)
    key = cv2.waitKey(1)
    if(key == 27):
        break
capture.release()
cv2.destroyAllWindows()
