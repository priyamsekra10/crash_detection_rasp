import cv2
import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import time
import pymongo
from pyfcm import FCMNotification
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from signal import signal, SIGPIPE, SIG_DFL
import datetime
import boto3
from tqdm import tqdm
import threading
import os
from dotenv import load_dotenv



load_dotenv()

# Access the environment variables
ACCESS_KEY_ID = os.getenv('ACCESS_KEY_ID')
SECRET_ACCESS_KEY = os.getenv('SECRET_ACCESS_KEY')

s3_client = boto3.client(
    's3',
    aws_access_key_id=ACCESS_KEY_ID,
    aws_secret_access_key=SECRET_ACCESS_KEY
)
# Initialize the S3 client
bucket_name = 'resq'  # replace with your actual bucket name

class ProgressPercentage(object):
    def __init__(self, filename):
        self._filename = filename
        self._size = float(s3_client.head_object(Bucket=bucket_name, Key=filename)['ContentLength'])
        self._seen_so_far = 0
        self._lock = threading.Lock()
        
    def __call__(self, bytes_amount):
        with self._lock:
            self._seen_so_far += bytes_amount
            percentage = (self._seen_so_far / self._size) * 100
            tqdm.write(f"{self._filename}  {self._seen_so_far}/{self._size}  ({percentage:.2f}%)")

def download_model_from_s3(model_key, local_path):
    # Using tqdm to display the progress bar
    with tqdm(total=int(s3_client.head_object(Bucket=bucket_name, Key=model_key)['ContentLength']), unit='B', unit_scale=True, desc=model_key) as pbar:
        s3_client.download_file(bucket_name, model_key, local_path, Callback=lambda bytes_transferred: pbar.update(bytes_transferred))

# Download the ONNX models from S3 and store the paths in variables
a = "car.h5"
download_model_from_s3("road_camera_models/car.h5", a)


# Constants
frame_interval = 0.1
IMG_SIZE = 224
email_id = 'priyam22rr@gmail.com'
last_emergency_response_time = 0

# MongoDB and Email Notification Functions
def emergency_response(email_id, crash_probability):
    client = pymongo.MongoClient("mongodb+srv://priyam:pqrs.123@cluster0.1uefwpt.mongodb.net/")
    db = client["car_crash"]
    collection = db["user_login"]
    
    def import_data_by_email(email_id):
        try:
            data_cursor = collection.find({"email_id": email_id})
            df = pd.DataFrame(list(data_cursor))

            if not df.empty:
                return df
            else:
                print("No data found for the provided email ID.")
                return None

        except Exception as e:
            print("Error occurred while importing data:", e)
            return None
        finally:
            client.close()

    email_id_to_search = email_id
    result_df = import_data_by_email(email_id_to_search)

    if result_df is not None:
        print("Data imported successfully for email:", email_id_to_search)
        
    for i in range(len(result_df['_id'])):
        fcm = result_df['fcm'][i]
        
    def notify_crash(fcm_token, crash_info):
        push_service = FCMNotification(api_key="AAAAucIfw-w:APA91bHy03w5pMy4AVf14qKy7M1Bw0JXMm4_A19r_KuY1viHVL3ky7wsqa34oaceDCTsQWaB5dGwa4gnDDqDnch9VvRjcl-fQw1YAY_WxNvhtigD5NGDftJEUSKJMp2ePWd3pQGS_UNm")

        message_title = "Crash Detected"
        message_body = "A crash was detected at location X."

        result = push_service.notify_single_device(
            registration_id=fcm_token, 
            message_title=message_title, 
            message_body=message_body, 
            data_message=crash_info
        )

    notify_crash(fcm_token=fcm, crash_info={
        'crash_time': '2021-07-11 14:30:00',
        'crash_location': 'Lat: 40.7128, Lon: 74.0060',
        'crash_severity': 'High',
        # ... any other data you want to send ...
    })

    print("Done")
    
    for i in range(len(result_df['r_email'])):
        receiver_email = result_df['r_email'][i]
        
    def send_email(sender_email, receiver_email, crash_probability, crash_location, crash_time):
        subject = 'Crash Detected'
        
        html_template = f"""
        <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    margin: 0;
                    padding: 0;
                }}
                .container {{
                    max-width: 600px;
                    margin: 20px auto;
                    padding: 20px;
                    background-color: #ffffff;
                    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
                    border-radius: 8px;
                }}
                .header {{
                    background-color: #4CAF50;
                    color: white;
                    padding: 10px 0;
                    text-align: center;
                    border-top-left-radius: 8px;
                    border-top-right-radius: 8px;
                }}
                .content {{
                    padding: 20px;
                }}
                .content h1 {{
                    color: #333333;
                }}
                .content p {{
                    color: #666666;
                    line-height: 1.6;
                }}
                .footer {{
                    text-align: center;
                    padding: 10px 0;
                    color: #aaaaaa;
                    font-size: 12px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Crash Alert!</h1>
                </div>
                <div class="content">
                    <h2>Details of the Incident:</h2>
                    <p><strong>Crash Probability:</strong> {crash_probability:.2f}</p>
                    <p><strong>Crash Location:</strong> {crash_location}</p>
                    <p><strong>Real-time at that moment:</strong> {crash_time}</p>
                </div>
                <div class="footer">
                    <p>This is an automated message. Please do not reply.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = receiver_email
        msg['Subject'] = subject
        msg.attach(MIMEText(html_template, 'html'))

        smtp_server = 'smtp.gmail.com'
        smtp_port = 587

        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, 'mlonvvatlfnilogu')
            server.send_message(msg)

    crash_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    crash_location = 'Lat: 26.9124, Lon: 75.7873'  # Replace with actual crash location if available
    send_email(sender_email=email_id, receiver_email=receiver_email, crash_probability=crash_probability, crash_location=crash_location, crash_time=crash_time)
    print("Email sent")

def can_call_emergency_response():
    global last_emergency_response_time
    current_time = time.time()
    return (current_time - last_emergency_response_time) >= 30 

def call_emergency_response(crash_probability):
    print("function called")
    print(crash_probability)
    global last_emergency_response_time
    emergency_response(email_id=email_id, crash_probability=crash_probability)
    last_emergency_response_time = time.time()
    print("=============-=-=-=======================-=-=-===================-=-=-===")

def load_crash_detection_model(model_path):
    custom_objects = {'KerasLayer': hub.KerasLayer}
    model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
    return model

def process_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = tf.image.resize(frame_rgb, size=[IMG_SIZE, IMG_SIZE])
    frame_tensor = tf.convert_to_tensor(frame_resized, dtype=tf.float32) / 255.0
    frame_batch = tf.expand_dims(frame_tensor, axis=0)
    return frame_batch

def predict_crash(model, frame_batch):
    prediction = model.predict(frame_batch)
    return prediction

def extract_frame_and_process(frame, model):
    frame_batch = process_frame(frame)
    prediction = predict_crash(model, frame_batch)
    print(prediction)
    crash_probability = prediction[0][1]
    
    crash_detected = False
    if crash_probability > 0.80:
        if can_call_emergency_response():
            call_emergency_response(crash_probability)
            crash_detected = True

    return crash_detected, crash_probability

def display_crash_probability(frame, crash_probability):
    text = f'Crash Probability: {crash_probability:.2f}'
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)


# Load the crash detection model
crash_detection_model_path = 'car.h5'
crash_detection_model = load_crash_detection_model(crash_detection_model_path)
frame_count = 0

# Open a connection to the laptop camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    # Process frame for crash detection
    crash_detected, crash_probability = extract_frame_and_process(frame, crash_detection_model)
    
    # Display crash probability on the frame
    display_crash_probability(frame, crash_probability)
    
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
signal(SIGPIPE, SIG_DFL)
