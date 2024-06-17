# The PiGuard - Video Surveillance with Raspberry Pi and Telegram Notifications

![mainpic](https://github.com/Chadoud/Visual-Detection-with-RasberryPi/assets/93930441/9278d049-53f9-4073-9e79-b1c1b27999ad)



## Overview
PiGuard is a project designed to use a Raspberry Pi and a camera to monitor a specified area, such as a waiting room. The system uses OpenCV and a pre-trained deep learning model to detect the presence of a person and sends notifications via Telegram when someone has been standing in the area for 3 seconds and can be adapted for different time laps and subjects such as animals, objects, ...

## Features
- Real-time detection using OpenCV and TensorFlow models.
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

c) Send a random message to your bot.


### 4. Obtain Your Telegram Chat ID

Run:
```bash
nano get_chat_id.py
```

Paste in the following code:
```python
import requests

# Change 'YOUR_BOT_TOKEN' by your given API token.
BOT_TOKEN = 'YOUR_BOT_TOKEN'
url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"

response = requests.get(url)
data = response.json()

print(data)
```
Press "ctlr+x" to save, then "y" and "Enter" to save changes

Run the script in a terminal:
```bash
python3 get_chat_id.py
```

Look for the chat object inside the message object in the JSON response. The id field within the chat object is your chat_id.

Keep YOUR_BOT_TOKEN (API Token) and YOUR_CHAT_ID (chat_id value) for the adaptation of the main "py_guard.py" file.


### 5. Download Model Files

Download the files:
```bash
cd Desktop
mkdir Object_Detection_Files
cd Object_Detection_Files
wget https://raw.githubusercontent.com/Chadoud/Visual-Detection-with-RasberryPi/main/coco.names
wget https://raw.githubusercontent.com/Chadoud/Visual-Detection-with-RasberryPi/main/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt
wget https://github.com/Chadoud/Visual-Detection-with-RasberryPi/raw/main/frozen_inference_graph.pb
wget https://raw.githubusercontent.com/Chadoud/Visual-Detection-with-RasberryPi/main/pi_guard.py
```


## Personalise your model in the "py_guard.py" file: 
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
