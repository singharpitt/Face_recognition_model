# Face Recognition Model

This project demonstrates basic face recognition using real-time data to provide accurate outputs.

## Overview

The Face Recognition Model utilizes computer vision techniques to recognize faces in real-time video streams or images. The model processes the input data, detects faces, and matches them against a database or predefined list of known faces. It then provides an output indicating whether a recognized face matches any known individuals.

## Features

- Real-time face detection and recognition.
- Matching detected faces against a database or known faces.
- Providing output with identification results.

## Technologies Used

- **Python**: Programming language used for development.
- **OpenCV**: Library for real-time computer vision.
- **Pillow**: Python Imaging Library for image processing.
- **Face Recognition**: Python library for face recognition tasks.
- **Haarcascade**: Classifier used for face detection.
- **LBPHFaceRecognizer**: Algorithm for face recognition.

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/face-recognition-model.git
   ```

2. Navigate to the project directory:

   ```
   cd face-recognition-model
   ```

3. Create and activate a virtual environment (optional but recommended):

   ```
   python -m venv venv
   source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
   ```

4. Install the required dependencies:

   ```
   pip install opencv-python
   pip install opencv-contrib-python
   pip install pillow
   pip install face-recognition
   ```

Alternatively, you can create a `requirements.txt` file and install all dependencies at once:

   ```
   pip install -r requirements.txt
   ```

   Contents of `requirements.txt`:
   ```
   opencv-python
   opencv-contrib-python
   pillow
   face-recognition
   ```

## Usage

1. **Face Dataset Collection**: 
   - Run the script to capture face images and create a dataset:
     ```
     python face_dataset.py
     ```

2. **Training the Model**: 
   - Train the face recognition model using the captured dataset:
     ```
     python training.py
     ```

3. **Real-time Face Recognition**: 
   - Run the script to start real-time face recognition:
     ```
     python face_recognition.py
     ```

   Ensure your camera is connected or provide a video stream as input. The application will detect faces in real-time and attempt to recognize them based on the pre-trained models or known face data.

## Contributing

Contributions are welcome! Fork the repository, make your changes, and submit a pull request.

