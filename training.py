import cv2
import numpy as np
from PIL import Image
import os

# Path for face image database
path = 'dataset'
trainer_path = 'trainer'

# Ensure trainer directory exists
if not os.path.exists(trainer_path):
    os.makedirs(trainer_path)

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

if not detector.load(cv2.samples.findFile("haarcascade_frontalface_default.xml")):
    print('--(!)Error loading face cascade')
    exit(0)

if not os.path.exists(path):
    print('--(!)Error: Dataset directory does not exist')
    exit(0)

if not os.listdir(path):
    print('--(!)Error: Dataset directory is empty')
    exit(0)

# Function to get the images and label data
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]     
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')  # convert to grayscale
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)

        for (x, y, w, h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)

    return faceSamples, ids

print("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
recognizer.write(f'{trainer_path}/trainer.yml')

# Print the number of faces trained and end program
print("\n [INFO] {0} faces trained.".format(len(np.unique(ids))))
