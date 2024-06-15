# PiGuard - Waiting Room Surveillance with Raspberry Pi

## Overview
PiGuard is a project designed to use a Raspberry Pi and a camera to monitor a specified area, such as a waiting room. The system uses OpenCV and a pre-trained deep learning model to detect the presence of a person and sends notifications via Telegram when someone has been standing in the area for 5 seconds and again after 5 minutes.

## Features
- Real-time person detection using OpenCV and TensorFlow models.
- Sends notifications via Telegram when a person is detected for specified durations.
- Configurable detection thresholds and notification intervals.

## Components
- Raspberry Pi (tested with Raspberry Pi 4 and 5), with at least 4GB of RAM and a Pi fan 
- Camera module compatible with Raspberry Pi
- Internet connection
- Telegram bot for notifications

## Setup Instructions

### 1. Setting Up the Raspberry Pi
1. Install Raspbian OS (Bullseye for Pi 4 or Bookwrom for Pi 5) with preferably a 64bit version.
2. Install your camera module and fan.
4. Ensure your Raspberry Pi is connected to the internet.

### 2. Install Required Libraries
Open a terminal and run the following commands to install the necessary libraries:

```bash
sudo apt-get update
sudo apt-get install python3-opencv python3-picamera2 python3-requests
```

Test your camera module by running: 

```bash
libcamera-still -o test.jpg
```


### 3. Set Up the Telegram Bot

a) Open Telegram and search for the "BotFather".

b) Start a chat with the BotFather and use the command /newbot to create a new bot.

c) Follow the instructions to set a name and username for your bot.

d) Once created, you will receive an API token. Keep this token secure.


### 4. Obtain Your Telegram Chat ID

```python
import requests

# Change 'YOUR_BOT_TOKEN' by your given API token.
BOT_TOKEN = 'YOUR_BOT_TOKEN'
url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"

response = requests.get(url)
data = response.json()

print(data)
```

Run the script in a terminal:
```bash
python3 get_chat_id.py
```

Look for the chat object inside the message object in the JSON response. The id field within the chat object is your chat_id.

Keep your YOUR_BOT_TOKEN and YOUR_CHAT_ID for the adaptation of the main "py_guard.py" file.


### 5. Download Model Files

Set up the files:
```bash
cd Desktop
mkdir Object_Detection
cd Object_Detection
```

Create and adapt the main file in the same "Object_Detection" directory:
```bash
nano py_guard.py
```

Paste the following script in the "py_guard.py" file:

```python
import cv2
import os
from picamera2 import Picamera2
import time
import requests

# Telegram bot API token and chat ID to replace with your values
TELEGRAM_BOT_TOKEN = 'YOUR_BOT_TOKEN'
TELEGRAM_CHAT_ID = 'YOUR_CHAT_ID'

# Function to send a Telegram message
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }
    requests.post(url, data=payload)

# Initialize variables for detection timing
detection_start_time = None
detected_for_5_seconds = False
detected_for_5_minutes = False

# Import Open-CV extra functionalities
users  = []
users.append(os.getlogin())

# This is to pull the information about what each object is called
classNames = []
classFile = f"/home/{users[0]}/Desktop/Object_Detection/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# This is to pull the information about what each object should look like
configPath = f"/home/{users[0]}/Desktop/Object_Detection/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = f"/home/{users[0]}/Desktop/Object_Detection/frozen_inference_graph.pb"

# Set up detection model
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

# Set up function to detect objects
def getObjects(img, thres, nms, draw=True, objects=["person"]):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    if len(objects) == 0: objects = classNames
    objectInfo = []
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects:
                objectInfo.append([box, className])
                if draw:
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    cv2.putText(img, classNames[classId - 1].upper(), (box[0] + 10, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(img, str(round(confidence * 100, 2)), (box[0] + 200, box[1] + 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    return img, objectInfo

# Main loop to capture video and detect objects
if __name__ == "__main__":
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
    picam2.start()
   
    while True:
        img = picam2.capture_array("main")
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        result, objectInfo = getObjects(img, 0.5, 0.2)
        
        person_detected = any(obj[1] == "person" for obj in objectInfo)
        
        if person_detected:
            if detection_start_time is None:
                detection_start_time = time.time()
            elapsed_time = time.time() - detection_start_time
            
            if elapsed_time >= 5 and not detected_for_5_seconds:
                send_telegram_message("Person detected for 5 seconds!")
                detected_for_5_seconds = True
        else:
            detection_start_time = None
            detected_for_5_seconds = False
            detected_for_5_minutes = False
        
        cv2.imshow("Output", img)
        k = cv2.waitKey(200)
        if k == 27:  # Esc key to stop
            picam2.stop()
            cv2.destroyAllWindows()
```



Download and paste the following files in the "Object_Detection" directory:

-ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt

-frozen_inference_graph.pb

-coco.names



Personalise your model: 
As the model is pretrained, you can find all the detectable objects in the coco.names file and
addapt the detection model by changing the "objects" value contained in the "getObjects" function. 

for example: 

def getObjects(img, thres, nms, draw=True, objects=["dog"])




### 6. Run the Script
Execute your script to start monitoring:

``` bash 
python3 pi_guard.py
```

### License
This project is licensed under the MIT License.
