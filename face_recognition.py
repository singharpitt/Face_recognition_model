import cv2
import numpy as np
import os

# Load the recognizer and face cascade
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

if not faceCascade.load(cv2.samples.findFile(cascadePath)):
    print('--(!)Error loading face cascade')
    exit(0)

if not os.path.exists('trainer/trainer.yml'):
    print('--(!)Error: Training file not found')
    exit(0)

# Font settings for text on the video feed
font = cv2.FONT_HERSHEY_TRIPLEX

# List of known names (should be dynamically fetched)
names = [0, 1, 2, 3, 'Z', 'W']

# Initialize and start real-time video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

# Define minimum window size to be recognized as a face
minW = 0.1 * cam.get(3)
minH = 0.1 * cam.get(4)

print("\n [INFO] Starting video stream. Press 'ESC' to exit.")

while True:
    ret, img = cam.read()
    if not ret:
        print("--(!)Error capturing video frame")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        id, confidence = recognizer.predict(gray[y:y + h, x:x + w])

        if confidence < 100:
            id = names[id] if id < len(names) else "Unknown"
            confidence_text = f"  {100 - confidence:.0f}%"
        else:
            id = "Unknown"
            confidence_text = f"  {100 - confidence:.0f}%"
        
        cv2.putText(img, str(id), (x + 5, y - 5), font, 1, (255, 255, 255), 2)
        cv2.putText(img, confidence_text, (x + 5, y + h - 5), font, 1, (255, 255, 0), 1)

    cv2.imshow('camera', img)

    k = cv2.waitKey(10) & 0xff
    if k == 27:  # Press 'ESC' to exit
        break

# Cleanup
print("\n [INFO] Exiting Program")
cam.release()
cv2.destroyAllWindows()
