from django.shortcuts import render
from django.http import StreamingHttpResponse
import cv2
from ultralytics import YOLO
from rajpofinalapp.tracker import Tracker
import random
from .models import FilesUpload,criminal_table,VideoUpload
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from tensorflow import keras
from tensorflow.keras.models import load_model
import easyocr
import matplotlib.pyplot as plt
from glob import glob
from random import sample
from PIL import ImageFont, ImageDraw, Image
import pandas as pd
import datetime
from collections import Counter
import os
from playsound import playsound
from django.core.files import File
# import pygame
# from pygame import mixer
from django.http import HttpResponse
from django.template import loader
from twilio.rest import Client
from django.conf import settings

stop=False
res = True
sir_stop=False
sirk_stop=False
reader = easyocr.Reader(['en'])
text=0
testlist = []
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
top_class="Nothing"
mobile="+919029171508"

obj_list = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "cow", "elephant",
    "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "sink", "bathtub", "shower", "towel",
    "shampoo", "soap", "toothbrush", "hair dryer", "makeup",
    "mirror", "book", "newspaper", "remote", "keyboard",
    "mouse", "cellphone", "laptop", "computer"
]

def vid_inp(request):
    # if request.method == "POST":
    #     file = request.FILES["file"]
    #     document = FilesUpload.objects.create(file=file)
    #     document.save()
    #     return video_feed(request, document.file.path)
    # else:
        return video_feed(request, 'rajpofinalapp/pik.mp4')


def video_feed(request, videopath):
    cap = cv2.VideoCapture(videopath)
    model = YOLO("yolov8n.pt")
    tracker = Tracker()
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(10)]
    detection_threshold = 0.5
    video_writer = cv2.VideoWriter(f"out_{timestamp}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 30, (IMG_SIZE_L, IMG_SIZE_B))
    video_path = f"out_{timestamp}.mp4"
    # video_path = f"out_{timestamp}.mp4"
    # video_file = open(video_path, "rb")
    # video_record = Video(title=f"Video {timestamp}", video_file=video_file)
    # video_record.save()

    print(f"Video Path: {video_path}")

    def gen_frames():
        global res
        global stop
        global sir_stop
        global sirk_stop
        global top_class
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if res==True :
                    # Resnet classification
                frame = cv2.resize(frame, (IMG_SIZE_L, IMG_SIZE_B))
                frame_features = feature_extractor.predict(frame[None, ...])
                frame_features = np.repeat(frame_features[None, ...], MAX_SEQ_LENGTH, axis=1)
                frame_mask = np.ones(shape=(1, MAX_SEQ_LENGTH), dtype="bool")

                probabilities = sequence_model.predict([frame_features, frame_mask])[0]
                top_class_index = np.argmax(probabilities)
                top_class = class_labels[top_class_index]
                confidence = probabilities[top_class_index] * 100
                # if confidence<=90:
                #     cv2.putText(frame, f"Class: None ({confidence:.2f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # else:
                # cv2.putText(frame, f"Class: {top_class} ({confidence:.2f}%)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                video_writer.write(frame)
                
                results = model(frame)
                classes_id=[]
                for result in results:
                        detections = []
                        for r in result.boxes.data.tolist():
                            x1, y1, x2, y2, score, class_id = r
                            x1 = int(x1)
                            x2 = int(x2)
                            y1 = int(y1)
                            y2 = int(y2)
                            class_id = int(class_id)
                            print(class_id)
                            if score > detection_threshold:
                                if class_id==43 and sirk_stop==False:
                                        sirk_stop=True 
                                        account_sid = settings.TWILIO_ACCOUNT_SID
                                        auth_token = settings.TWILIO_AUTH_TOKEN
                                        client = Client(account_sid, auth_token)
                                        client.messages.create(
                                            body="Alert knife detected",
                                            from_="+12055905182",
                                            to=mobile
                                        )
                                        print("siren started")

                                        playsound("siren.mp3")
                            # detections.append([x1, y1, x2, y2, score])
                            # classes_id.append(class_id)
                # update_anomaly_alert(top_class)

            if sir_stop==False and res==False:
                    account_sid = settings.TWILIO_ACCOUNT_SID
                    auth_token = settings.TWILIO_AUTH_TOKEN
                    client = Client(account_sid, auth_token)
                    client.messages.create(
                        body=f"Alert {top_class} detected",
                        from_="+12055905182",
                        to=mobile
                    )
                    print("siren started")

                    playsound("siren.mp3")
                    sir_stop=True

                    # print(f"Video Path: {video_path}")


            if confidence >= 90 or res==False:
                global text
                global testlist
                # Continue with YOLO and DeepSORT tracking
                res=False
                #frame = cv2.resize(frame, (IMG_SIZE_L, IMG_SIZE_B))
                frame = cv2.resize(frame, (IMG_SIZE_L, IMG_SIZE_B))
                results = model(frame)
                classes_id=[]
                for result in results:
                    detections = []
                    for r in result.boxes.data.tolist():
                        x1, y1, x2, y2, score, class_id = r
                        x1 = int(x1)
                        x2 = int(x2)
                        y1 = int(y1)
                        y2 = int(y2)
                        class_id = int(class_id)
                        print(class_id)
                        if score > detection_threshold:
                            detections.append([x1, y1, x2, y2, score])
                            classes_id.append(class_id)
                #class_names=[obj_list[i]for i in classes_id]

                tracker.update(frame, detections)
                
                for track in tracker.tracks:
                    bbox = track.bbox
                    x1, y1, x2, y2 = bbox
                    track_id = track.track_id
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (colors[track_id % len(colors)]), 3)
                    #cv2.putText(frame,' '.join(class_names), (int(x1), int(y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                if stop==False:
                    results = reader.readtext(frame)
                    for (bbox, text, prob) in results:
                                (top_left, top_right, bottom_right, bottom_left) = bbox
                                top_left = tuple(map(int, top_left))
                                bottom_right = tuple(map(int, bottom_right))
                                cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
                                cv2.putText(frame, text, top_left, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                                testlist.append(text)
                
                element_counts = Counter(testlist)

# Find the element with the maximum count
                max_element = max(element_counts, key=element_counts.get)

                if len(testlist)==4 and stop==False:
                        matching_rows = criminal_table.objects.filter(criminal_id=max_element)
                        if matching_rows.exists():
                            existing_excel_filename = 'existing_data.xlsx'
                            df_existing = pd.read_excel(existing_excel_filename)
                            data = {'name': [], 'criminal_id': [], 'image': [],'case': [],'Inperiod': [],'area': [],'timestamp' :[] }  # Add other fields as needed
                            for row in matching_rows:
                                data['name'].append(row.name)
                                data['criminal_id'].append(row.criminal_id)
                                data['image'].append(row.image)
                                data['case'].append(row.case)
                                data['Inperiod'].append(row.Inperiod)
                                data['area'].append(row.area)
                            data['timestamp'].append(timestamp)
                            df_new = pd.DataFrame(data)

            # # Save DataFrame to Excel file
                            df_updated = pd.concat([df_existing, df_new], ignore_index=True)

            # # Save the updated DataFrame back to the Excel file
                            df_updated.to_excel(existing_excel_filename, index=False)

                
        #                 for row in matching_rows:
        #                     print(row.__dict__)

                if len(testlist)>=4:
                    stop=True
                

                print(testlist)  
                video_writer.write(frame)


            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    
        if stop and not VideoUpload.objects.filter(title=f"Video {timestamp}").exists():
            with open(video_path, 'rb') as video_file:
                video_record = VideoUpload(title=f"Video {timestamp}")
                video_record.video_file.save(f"{timestamp}.mp4", File(video_file))
                video_record.save()

    return StreamingHttpResponse(gen_frames(), content_type="multipart/x-mixed-replace;boundary=frame")


# Resnet code
IMG_SIZE_L = 480
IMG_SIZE_B = 640
MAX_SEQ_LENGTH = 100
NUM_FEATURES = 2048

# Load the trained Resnet model
sequence_model = keras.models.load_model('rajpofinalapp/saved_model')

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y: start_y + min_dim, start_x: start_x + min_dim]

def build_feature_extractor():
    feature_extractor = keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(640, 480, 3),
    )
    preprocess_input = keras.applications.inception_v3.preprocess_input

    inputs = keras.Input((640, 480, 3))
    preprocessed = preprocess_input(inputs)

    outputs = feature_extractor(preprocessed)
    return keras.Model(inputs, outputs, name="feature_extractor")

feature_extractor = build_feature_extractor()

# Dummy DataFrame for class labels
class_labels = ['Fencing', 'Punch', 'RockClimbingIndoor', 'RopeClimbing', 'WallPushUps']
label_processor = keras.layers.experimental.preprocessing.StringLookup(
    num_oov_indices=0, vocabulary=class_labels
)


from django.shortcuts import render

from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login
from .models import CustomUser

from django.shortcuts import render

def home(request):
    # ...
    template = loader.get_template('home.html')
    context = {
        'anomaly_class': top_class,
    }
    rendered = template.render(context, request)
    return HttpResponse(rendered)

def analysis(request):
    return render(request,"analysis.html") 

def history(request):
    return render(request,"history.html")

def signup(request):
    if request.method == 'POST':
        full_name = request.POST.get('full_name')
        phone = request.POST.get('phone')
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = CustomUser.objects.create_user(
            full_name=full_name,
            phone=phone,
            username=username,
            password=password
        )

        login(request, user)

        return redirect('home') 
    else:
        return render(request, 'login.html')

def signin(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)

        if user is not None:
            login(request, user)
            return redirect('home') 
        else:
            return render(request, 'login.html', {'error_message': 'Invalid username or password'})

    return render(request, 'login.html')

# views.py
from django.http import JsonResponse

def update_anomaly(request):
    global top_class
    # print(top_class)

    context = {'anomaly_class': top_class}
    return JsonResponse(context)
