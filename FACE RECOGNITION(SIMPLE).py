import cv2
import face_recognition
import numpy as np

# Load known face and encode it
known_image = face_recognition.load_image_file("known.jpg")  # Replace with your image
known_encoding = face_recognition.face_encodings(known_image)[0]

# Start webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect faces and encode them
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        match = face_recognition.compare_faces([known_encoding], face_encoding)[0]
        name = "Matched" if match else "Unknown"
        
        # Draw a rectangle and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Show the webcam feed
    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
