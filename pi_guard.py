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
classFile = f"/home/{users[0]}/Desktop/Object_Detection_Files/coco.names"
with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# This is to pull the information about what each object should look like
configPath = f"/home/{users[0]}/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = f"/home/{users[0]}/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

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
            
            if elapsed_time >= 3 and not detected_for_3_seconds:
                send_telegram_message("Person detected")
                detected_for_3_seconds = True
        else:
            detection_start_time = None
            detected_for_3_seconds = False
        
        cv2.imshow("Output", img)
        k = cv2.waitKey(200)
        if k == 27:  # Esc key to stop
            picam2.stop()
            cv2.destroyAllWindows()
