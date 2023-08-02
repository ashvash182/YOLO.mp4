from flask import Flask, jsonify, request
from flask_cors import CORS
import requests

import os
from io import BytesIO
from PIL import Image

from celery import Celery
from celery.result import AsyncResult
import time

import torch, torchvision
import cv2
import numpy as np

# Outputs individual video frames to a folder (Not used)
def save_video_frames_to_folder(url):
    vcap = cv2.VideoCapture(url)
    
    frames = []

    count = 0
    while(vcap.isOpened()):
        # Capture frame-by-frame
        ret, frame = vcap.read()

        if not ret:
            break
        # Convert the frame to a Pillow image
        with BytesIO() as output:
            Image.fromarray(frame).save(output, format='JPEG')
            image = Image.open(output)
            frames.append(image)
            image.save(output, format='JPEG')
        image = Image.fromarray(frame)
        image.save(f"frames/frame_{count}.jpg")
        count += 1

    return frames