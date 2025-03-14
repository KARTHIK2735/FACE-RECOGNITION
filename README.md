# FACE-RECOGNITION
A Face recognition project in Python using Face Recognition Library. This project leverages  OpenCV for image processing and NumPy for efficient numerical operations. The system detects and  recognizes faces by encoding facial features and comparing them with stored embeddings. 
This project is a simple face recognition system using OpenCV and face_recognition libraries. It allows you to encode known faces, recognize faces in images, and recognize faces in a live webcam feed.

## Features
- Encode known faces from a directory
- Recognize faces in a given image
- Perform live face recognition using a webcam

## Requirements
Ensure you have the following dependencies installed before running the project:

- Python 3.11 ( for cv2  to work properly)
- OpenCV (`cv2`)
- face_recognition
- NumPy

### Install Dependencies
Run the following command to install the required libraries:
```sh
pip install opencv-python face-recognition numpy
```

## Directory Structure
```
face_recognition_project/
│── known_faces/       # Folder containing subfolders for each person with their images
│── unknown_faces/     # Folder for images to be recognized
│── face_recognition.py # Main script
│── face_encodings.npy  # Stored face encodings (generated after encoding faces)
│── face_names.npy      # Stored names corresponding to the encodings
```

## Usage
Run the script and follow the on-screen instructions:

```sh
python face_recognition.py
```

### Options:
1. **Encode Known Faces**
   - Place images of known individuals in the `known_faces/` directory. Each person should have their own subfolder with their images.
   - Select option `1` to process and encode faces. Encodings will be saved in `face_encodings.npy` and `face_names.npy`.

2. **Recognize Faces in Image**
   - Select option `2` and provide the path to an image file.
   - The system will attempt to recognize faces based on stored encodings and display the image with labeled faces.

3. **Recognize Faces in Webcam**
   - Select option `3` to start real-time face recognition via webcam.
   - Press `q` to exit the webcam feed.

## Notes
- If no known faces are encoded, options 2 and 3 will not function properly.
- Ensure images are clear and well-lit for better recognition accuracy.
- 
## Author
`KARTHIK`

