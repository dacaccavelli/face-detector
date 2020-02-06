# -*- coding: utf-8 -*-
"""
Daniel Caccavelli

Person Tracker and Handwriting Detector??
"""
import numpy as np
import cv2
import scipy.io
from keras.applications import inception_v3
import random

# Variable holding imported xmls for detecting facial features.
face_cascade = cv2.CascadeClassifier(r".\haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(r".\haarcascade_eye.xml")

# Activating camera
camera = cv2.VideoCapture(0)
camera_height = 1000
hatBool = False


def givehat(frame):
    """Function for attaching the hats to the frame in the appropriate
    positions.
    """
    for (x, y, w, h) in faces:
        # Checks if hat will remain on screen.
        if x - w > 0 and y - w > 0:
            choice = random.randint(2, 2)
            hat = cv2.imread(
                r'.\Hats\hat%s.png' % (choice))
            # Resizes the hat proportional to face size.
            resized_hat = cv2.resize(hat, (w, w))
            # Masks the selected image above the face.
            frame[y-w:y, x:x+w] = resized_hat
    return frame


while(True):

    # Read a new frame.
    _, frame = camera.read()

    # Flip the frame.
    frame = cv2.flip(frame, 1)
    # Rescaling camera output.
    aspect = frame.shape[1]/float(frame.shape[0])
    res = int(aspect * camera_height)  # landscape orientation - wide image
    frame = cv2.resize(frame, (res, camera_height))

    # Face + eye detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 2)

    # Checks to see if user wants to apply hats or draw rectangles.
    if hatBool:
        frame = givehat(frame)
    else:
        # Drawing outline around facial features.
        for (x, y, w, h) in faces:
            # Drawing outline around detected faces.
            img = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)

            # Converts image to grayscale for ease in detecting eyes.
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                # Drawing outline around detected eyes.
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh),
                              (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Capturing frames", frame)

    key = cv2.waitKey(1)

    # Quit camera if 'q' is pressed
    # Switches from rectangles to giving hats if 'h' is pressed.
    if key & 0xFF == ord("q"):
        break
    elif key & 0xFF == ord("h"):
        hatBool = not(hatBool)

camera.release()
