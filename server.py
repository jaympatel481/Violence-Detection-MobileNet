from flask import Flask, Response, render_template
import cv2
import numpy as np
import argparse
import pickle
import os
import time
from keras.models import load_model
from collections import deque
from playsound import playsound

import notify

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

app = Flask(__name__, template_folder='templates')

# for local webcam we used 0, we can use IP CCTV camera using rtsp protocol
camera = cv2.VideoCapture(0)

# generate frame by frame from camera
def gen_frames():
    # load mobilenetv2 model to predict frames
    model = load_model('./Model/model.h5')
    model.load_weights('./Model/ModelWeights.h5')
    Q = deque(maxlen=128)
    (W, H) = (None, None)
    count = 0
    Violence = False

    while True:
        if Violence is True:
            playsound('./static/Alarm.mp3')
        # capture frame-by-frame
        success, frame = camera.read()
        if not success:
            blank_frame = np.zeros(shape=[512, 512, 3], dtype=np.uint8)
            output = blank_frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (128, 128)).astype("float32")
            frame = frame.reshape(128, 128, 3) / 255
            Q.append(frame)

            # perform prediction averaging over the current history of
            # previous predictions
            results = np.array(Q).mean(axis=0)
            text_color = (0, 0, 0) # default : green

            text = "Camera/Server Service Needed".format(label)
            FONT = cv2.FONT_HERSHEY_SIMPLEX

            cv2.putText(output, text, (35, 50), FONT,1.25, text_color, 3)
            ret, buffer = cv2.imencode('.jpg', output)
            output = buffer.tobytes()
            yield (b'--frame\r\n'+b'Content-Type: image/jpeg\r\n\r\n' + output + b'\r\n')

        else:
            # if the frame dimensions are empty, grab them
            if W is None or H is None:
                (H, W) = frame.shape[:2]

            # clone the output frame, then convert it from BGR to RGB
            # ordering, resize the frame to a fixed 128x128, and then
            # perform mean subtraction
            output = frame.copy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (128, 128)).astype("float32")
            frame = frame.reshape(128, 128, 3) / 255

            # make predictions on the frame and then update the predictions
            # queue
            preds = model.predict(np.expand_dims(frame, axis=0))[0]
            # print("preds",preds)
            Q.append(preds)

            # perform prediction averaging over the current history of
            # previous predictions
            results = np.array(Q).mean(axis=0)
            i = (preds > 0.58)[0]
            label = i
            text_send = "Non-Sever Violence"

            if preds>0.7:
                text_send = "Sever Violence"
                notify.notify()

            if label:
                Violence = True
                text_color = (0, 0, 255) # red
                text = text_send
                FONT = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(output, text, (35, 50), FONT,1, text_color, 3)
            else:
                Violence = False
                text_color = (0, 255, 0)
                text = "No Violence"
                FONT = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(output, text, (35, 50), FONT,1.25, text_color, 3)

            ret, buffer = cv2.imencode('.jpg', output)
            output = buffer.tobytes()
            yield (b'--frame\r\n'+b'Content-Type: image/jpeg\r\n\r\n' + output + b'\r\n')


@app.route('/api/62b5129901bbada7a7be3ff40b608ae4')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/cbf372ac6a78f7207a67ddb667abf076')
def notify_to_admin():
    notify.notify()
    return ('', 204)


@app.route('/')
def index():
    return render_template('video.html')


if __name__ == '__main__':
    app.run(host = '127.0.0.1', port = 8889, debug=True)
