from flask import Flask, request

from io import BytesIO
from PIL import Image
import urllib.request

from celery import Celery, current_task
from celery.result import AsyncResult
import subprocess

import torch
import cv2
import numpy as np
from sort.sort import Sort

app = Flask(__name__)
CELERY_BROKER_URL = 'redis://127.0.0.1:6379'
CELERY_RESULT_BACKEND = 'redis://127.0.0.1:6379'

model = torch.hub.load("ultralytics/yolov5", "yolov5s")
jobs = []

def get_celery_app_instance(app):
    # Intialize celery instance from Flask app.
    celery = Celery(
        app.import_name,
        backend=CELERY_BROKER_URL,
        broker=CELERY_BROKER_URL,
        CELERY_TASK_TRACK_STARTED=True,
        result_backend='redis://'
    )
    celery.conf.update(app.config)

    class ContextTask(celery.Task):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return self.run(*args, **kwargs)

    celery.Task = ContextTask
    return celery

celery = get_celery_app_instance(app)

@celery.task    
def process_video(source_name, source_url):
    # Celery task to do ML processing on the specified video.
    current_task.update_state(state="STARTED")
    # Make API request here
    if (source_url==''):
        return ('No url provided.')
    
    video_data = get_video_from_url(source_url)
    
    if (len(video_data)==0):
        return ('Unable to get video from url.')
    
    tracker = Sort()

    objects = {}
    for i in enumerate(video_data):
        frame = video_data[i[0]]
        
        detections = model(frame)
            
        curr_boxes = []
        objs = []
        for row in detections.xyxy[0]:
            x1, y1, x2, y2, score = float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])
            objs.append(row[5])
            curr_boxes.append([x1, y1, x2, y2])
        objs = np.array(objs)
        
        if (curr_boxes == []):
            tracks = tracker.update(np.empty((0, 5)))
        else:
            tracks = tracker.update(np.array(curr_boxes))
        if (tracks is None):
            continue
        for t in enumerate(tracks):
            track = t[1]
            obj_class = int(objs[t[0]])
            object_id = int(track[4])
            if object_id not in objects:
                objects[object_id] = {"class": detections.names[obj_class], "positions": []}
            x1, y1, x2, y2 = map(int, track[:4])
            objects[object_id]["positions"].append({"frame_number": i[0], "x1": x1, "x2": x2, "y1": y1, "y2": y2})
    return objects

@app.route('/push', methods=['POST'])   
def enqueue_push_request():
    # Add push request to a queue where each task will be processed one-by-one.
    source_name = request.args.get('source_name')
    source_url = request.args.get('source_url')
    task = process_video.delay(source_name, source_url)

    jobs.append(task.id)

    return {'id' : task.id}

@app.route('/status', methods=['GET'])
def get_job_status():
    # Provide job status from Celery queue.
    task_id = request.args.get('id')
    if (task_id not in jobs):
        return {'status' : 'job does not exist'}
    # Celery
    status = celery.AsyncResult(task_id).state
    
    if status == 'PENDING':
        return {'status' : 'queued'}
    elif status == 'STARTED':
        return {'status' : 'processing'}
    elif status == 'SUCCESS':
        return {'status' : 'finished'}
    elif status == 'FAILURE':
        return {'status' : 'failed'}

@app.route('/query', methods=['GET'])
def query_func():
    # Will return objects detected in the video if the specified task has been completed.
    task_id = request.args.get('id')
    if (task_id not in jobs):
        return ('Job does not exist.')
    # Celery
    result = celery.AsyncResult(task_id)
    if (result.ready()):
        return result.get()
    else:
        return ('Task still running...')

@app.route('/list', methods=['GET'])
def query():
    return jobs

def get_video_from_url(url):
    # Retrieve individual video frames from URL
    try:
        vcap = cv2.VideoCapture(url)
    except:
        return ('Unable to grab video from specified link.')
    
    frames = []

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

    return frames