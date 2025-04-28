# Ligpit-Pang-Higpit
An AI-Integrated Bolts and Nuts Robotic Sorting System using Deep Learning and Computer Vision
![image](https://github.com/user-attachments/assets/5cd1f5ce-de7f-4c69-9bcb-5bd4fab0c5e4)

![image](https://github.com/user-attachments/assets/83bf859d-6b67-4856-9899-e157e1e520b3)

---

# 📖 Full Tutorial: Building a Bolt and Nut Detection Robotic Arm using Computer Vision, Roboflow, Python, and Arduino

---

## Overview
In this project, we will build a **vision-guided robotic arm** that **detects bolts and nuts** through a **webcam** using **AI object detection**, and automatically **moves** to sort them based on what is seen.  

We'll use:
- **Computer Vision (Roboflow)** for object recognition,
- **Python** for handling the webcam, image processing, and sending detection results,
- **Arduino** to control a **robotic arm** using **servo motors**.

> **Goal**: Real-time bolt/nut detection ➔ Move robotic arm to sort.

---

## Materials Needed

| Quantity | Item |
|:--------:|:----:|
| 1 | Arduino Uno or Mega |
| 1 | PC or Laptop (with Python installed) |
| 1 | USB Cable for Arduino |
| 1 | 6x Servo Motors (e.g., SG90, MG996R, or similar) |
| 1 | External 5V Power Supply (for servos) |
| 1 | Breadboard & Jumper Wires |
| 1 | Webcam (USB) |
| 1 | Robotic Arm Frame (DIY or Kit) |
| 1 | Roboflow Account (for object detection API) |
| - | Basic nuts and bolts (for testing) |

---

## System Architecture

```plaintext
[Webcam] 
   ↓ 
[Python Program]
   - Capture frame
   - Send frame to Roboflow API
   - Detect bolt/nut
   - Send command via Serial
   ↓ 
[Arduino]
   - Read command (bolt/nut/none)
   - Move robotic arm accordingly
   - Signal "done" when finished
```

## Block Diagram

```
![image](https://github.com/user-attachments/assets/965aaaca-f8b8-42b5-a2ac-9f0b301f2cba)

```

## System FlowChart

```
![image](https://github.com/user-attachments/assets/516f81de-b2c7-4398-82c5-c8e3ddc39c4c)

```

---

## Wiring Diagram

| Arduino Pin | Connected To |
|:-----------:|:------------:|
| 3 | Waist Servo Signal |
| 5 | Shoulder Servo Signal |
| 6 | Elbow Servo Signal |
| 9 | Wrist Pitch Servo Signal |
| 10 | Wrist Roll Servo Signal |
| 11 | Gripper Servo Signal |
| GND | Servo Power Ground |
| 5V | Servo Power +5V (small servos) or external power |

---
> **Important**: If you are using **power-hungry servos**, connect the servos' VCC to an **external 5V power supply**, **NOT directly** to the Arduino 5V pin.

---

## Software Installation

### 1. Python Requirements
- Python 3.x
- Install the required libraries:

```bash
pip install opencv-python pyserial requests
```

### 2. Arduino IDE
- Download and install the Arduino IDE from [https://www.arduino.cc/en/software](https://www.arduino.cc/en/software)

- Install the Servo library (already included by default).

---

## Setting Up Roboflow
1. Go to [https://roboflow.com/](https://roboflow.com/)
2. Create an account.
3. Create a new project.
4. Upload images of bolts and nuts.
5. Label them properly (`bolt`, `nut`).
6. Train a model.
7. Get your API key and model endpoint URL.
   - Example API Endpoint used in this tutorial:
     ```
     https://detect.roboflow.com/bolts-and-nuts-vhkyw/8?api_key=YOUR_API_KEY
     ```

---

## Python Code 

```python
import cv2
import base64
import requests
import json
import serial
import time

# Roboflow API URL and your API key
ROBOFLOW_API_URL = "https://detect.roboflow.com/bolts-and-nuts-vhkyw/8?api_key=eDUqvZMpJlgRmGks0Szk"

# Establish a serial connection (replace 'COM3' with your Arduino's port)
ser = serial.Serial('COM3', 9600, timeout=1)
time.sleep(2)  # Wait for the connection to be established

def get_detections_from_roboflow(image):
    # Encode image to base64
    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')

    # Send the image to the Roboflow API
    response = requests.post(ROBOFLOW_API_URL, data=image_base64, headers={'Content-Type': 'application/x-www-form-urlencoded'})

    # Parse the JSON response
    detections = json.loads(response.text)
    return detections

def draw_detections(image, detections):
    # Draw bounding boxes on the image
    for detection in detections.get('predictions', []):
        x1, y1 = int(detection['x'] - detection['width'] / 2), int(detection['y'] - detection['height'] / 2)
        x2, y2 = int(detection['x'] + detection['width'] / 2), int(detection['y'] + detection['height'] / 2)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, detection['class'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 2)
    return image

def send_command(command):
    ser.write(f"{command}\n".encode())

def wait_for_arduino():
    while True:
        if ser.in_waiting > 0:
            response = ser.readline().decode().strip()
            if response == "done":
                break

# Open the webcam
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

try:
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Get detections from Roboflow
        detections = get_detections_from_roboflow(frame)

        # Check for specific objects and send commands to Arduino
        if detections.get('predictions'):
            detected_classes = {detection['class'] for detection in detections['predictions']}
            if 'bolt' in detected_classes:
                send_command('bolt')
                wait_for_arduino()
            elif 'nut' in detected_classes:
                send_command('nut')
                wait_for_arduino()
        else:
            send_command('none')
            wait_for_arduino()

        # Draw detections on the frame
        annotated_frame = draw_detections(frame, detections)

        # Display the resulting frame
        cv2.imshow('YOLOv8 Webcam', annotated_frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            send_command('quit')
            wait_for_arduino()
            break

except KeyboardInterrupt:
    print("Program terminated.")

# When everything is done, release the capture and close windows
cap.release()
cv2.destroyAllWindows()
ser.close()  # Close the serial connection
```

---

## Arduino Code (Note: Calibrate the movements depending on how you set the motors because it varies, also becareful on calibrating to avoid breaking your motors)

```cpp
#include <Servo.h>

Servo myservowaist;
Servo myservoshoulder;
Servo myservoelbow;
Servo myservowristpitch;
Servo myservowristroll;
Servo myservogrip;

// Array to store current angles for each servo
int currentAngles[6] = {95, 40, 0, 170, 0, 80};

String detectedItem;

void setup() {
  Serial.begin(9600);

  myservowaist.attach(3);
  myservoshoulder.attach(5);
  myservoelbow.attach(6);
  myservowristpitch.attach(9);
  myservowristroll.attach(10);
  myservogrip.attach(11);
  delay(10);
  defaultPos();
  delay(2000);
}

void loop() {
  if (Serial.available() > 0) {
    detectedItem = Serial.readStringUntil('\n');
    if (detectedItem == "bolt") {
      boltSection();  
      defaultPos();
    } else if (detectedItem == "nut") {
      nutSection();  
      defaultPos();
    } else if (detectedItem == "none") {
      defaultPos();
    }
    Serial.println("done");  
  }
}

void moveServoSlowly(Servo servo, int endAngle, int step, int index) {
  int startAngle = currentAngles[index];
  int increment = (startAngle < endAngle) ? step : -step;

  while ((increment > 0 && startAngle <= endAngle) || (increment < 0 && startAngle >= endAngle)) {
    servo.write(startAngle);
    delay(30); 
    startAngle += increment;
  }
  servo.write(endAngle); 
  currentAngles[index] = endAngle; 
}

void defaultPos() {
  moveServoSlowly(myservowristroll, 0, 2, 4);
  delay(100);
  moveServoSlowly(myservoelbow, 0, 2, 2);
  delay(100);
  moveServoSlowly(myservowristpitch, 170, 2, 3);
  delay(100);
  moveServoSlowly(myservoshoulder, 40, 2, 1);
  delay(100);
  moveServoSlowly(myservowaist, 95, 2, 0);
  delay(100);
  moveServoSlowly(myservogrip, 80, 2, 5);
  delay(100);
}

void grabObject() {
  moveServoSlowly(myservoshoulder, 60, 2, 1);
  delay(500);
  moveServoSlowly(myservogrip, 50, 2, 5);
  delay(500);
  moveServoSlowly(myservoshoulder, 95, 2, 1);
  delay(500);
  moveServoSlowly(myservoelbow, 25, 2, 2);
  delay(500);
  moveServoSlowly(myservowristroll, 90, 2, 4);
  delay(100);
  moveServoSlowly(myservoshoulder, 110, 2, 1);
  delay(500);
  moveServoSlowly(myservoelbow, 45, 2, 2);
  delay(500);
  moveServoSlowly(myservoshoulder, 130, 2, 1);
  delay(500);
  moveServoSlowly(myservoelbow, 50, 2, 2);
  delay(500);
  moveServoSlowly(myservoshoulder, 141, 2, 1);
  delay(500);
  moveServoSlowly(myservogrip, 80, 2, 5);
  delay(500);
  moveServoSlowly(myservoshoulder, 130, 2, 1);
  delay(500);
  moveServoSlowly(myservoelbow, 45, 2, 2);
  delay(500);
  moveServoSlowly(myservoshoulder, 110, 2, 1);
  delay(500);
  moveServoSlowly(myservoelbow, 25, 2, 2);
  delay(500);
  moveServoSlowly(myservowristpitch, 140, 2, 3);
  delay(500);
}

void releaseObject() {
  moveServoSlowly(myservogrip, 50, 2, 5);
  delay(100);
  moveServoSlowly(myservogrip, 80, 2, 5);
  delay(500);
  moveServoSlowly(myservowristpitch, 110, 2, 3);
  delay(500);
  moveServoSlowly(myservoshoulder, 90, 2, 1);
  delay(100);
  defaultPos();
}

void boltSection() {
  grabObject();
  moveServoSlowly(myservowaist, 80, 2, 0);
  delay(100);
  moveServoSlowly(myservowaist, 55, 2, 0);
  delay(100);
  releaseObject();
}

void nutSection() {
  grabObject();
  moveServoSlowly(myservowaist, 105, 2, 0);
  delay(100);
  moveServoSlowly(myservowaist, 135, 2, 0);
  delay(100);
  releaseObject();
}
```

---

## Final Testing

1. Connect Arduino to your PC.
2. Upload the Arduino code.
3. Run the Python script.
4. Place nuts and bolts under the webcam.
5. The arm should automatically grab and sort based on what was detected!

---

## Conclusion

This system integrates **AI + Computer Vision + Robotics** beautifully:
- Roboflow handles object detection.
- Python handles vision processing and communication.
- Arduino executes the mechanical movement.

---
