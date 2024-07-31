import cv2
import os

# Ensure the dataset directory exists
if not os.path.exists('dataset'):
    os.makedirs('dataset')

# Initialize webcam
cam = cv2.VideoCapture(0)
cam.set(3, 640)  # set video width
cam.set(4, 480)  # set video height

# Load Haar cascade for face detection
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
if not face_detector.load(cv2.samples.findFile('haarcascade_frontalface_default.xml')):
    print('--(!)Error loading face cascade')
    exit(0)
if not cam.isOpened():
    print('--(!)Error opening video capture')
    exit(0)

# Get user ID
face_id = input('\n enter user id fo the user :- ')

print("\n [INFO] Initializing face capture....")
print("\n [INFO] Press 'ESC' to quit or wait for 200 samples to be captured.")

count = 0

while True:
    ret, img = cam.read()
    if not ret:
        print("--(!)Error capturing image")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        count += 1

        # Save the captured image into the datasets folder
        cv2.imwrite("dataset/User." + str(face_id) + '.' + str(count) + ".jpg", gray[y:y+h, x:x+w])

        cv2.imshow('image', img)

    k = cv2.waitKey(100) & 0xff
    if k == 27:  # Press 'ESC' to quit
        break
    elif count >= 200:  # Take 200 face samples and stop the video
        break

# Cleanup
print("\n [INFO] Exiting Program")
cam.release()
cv2.destroyAllWindows()
