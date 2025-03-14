import cv2
import face_recognition
import numpy as np
import os

# Directory to store known faces
KNOWN_FACES_DIR = "known_faces"
UNKNOWN_FACES_DIR = "unknown_faces"
ENCODINGS_FILE = "face_encodings.npy"
NAMES_FILE = "face_names.npy"

# Function to load and encode known faces
def encode_known_faces():
    known_encodings = []
    known_names = []
    
    for name in os.listdir(KNOWN_FACES_DIR):
        person_dir = os.path.join(KNOWN_FACES_DIR, name)
        
        if not os.path.isdir(person_dir):
            continue
        
        for image_name in os.listdir(person_dir):
            image_path = os.path.join(person_dir, image_name)
            image = face_recognition.load_image_file(image_path)
            encodings = face_recognition.face_encodings(image)
            
            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(name)
                print(f"Encoded {name} from {image_name}")
    
    np.save(ENCODINGS_FILE, known_encodings)
    np.save(NAMES_FILE, known_names)
    print("Face encodings saved!")

# Function to recognize faces in a new image
def recognize_faces_in_image(image_path):
    # Load known face encodings
    if not os.path.exists(ENCODINGS_FILE) or not os.path.exists(NAMES_FILE):
        print("No known faces found! Please encode known faces first.")
        return
    
    known_encodings = np.load(ENCODINGS_FILE, allow_pickle=True)
    known_names = np.load(NAMES_FILE, allow_pickle=True)

    # Load new image
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    # Convert to OpenCV format
    image_cv2 = cv2.imread(image_path)
    
    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"
        
        if True in matches:
            best_match_index = matches.index(True)
            name = known_names[best_match_index]
        
        # Draw rectangle and label
        cv2.rectangle(image_cv2, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image_cv2, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Show image
    cv2.imshow("Face Recognition", image_cv2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to recognize faces in a live webcam feed
def recognize_faces_in_webcam():
    # Load known face encodings
    if not os.path.exists(ENCODINGS_FILE) or not os.path.exists(NAMES_FILE):
        print("No known faces found! Please encode known faces first.")
        return
    
    known_encodings = np.load(ENCODINGS_FILE, allow_pickle=True)
    known_names = np.load(NAMES_FILE, allow_pickle=True)

    # Start webcam
    video_capture = cv2.VideoCapture(0)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Unknown"
            
            if True in matches:
                best_match_index = matches.index(True)
                name = known_names[best_match_index]
            
            # Draw rectangle and label
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow("Webcam Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

# Main menu
if __name__ == "__main__":
    print("Face Recognition System")
    print("1. Encode Known Faces")
    print("2. Recognize Faces in Image")
    print("3. Recognize Faces in Webcam")
    
    choice = input("Choose an option (1/2/3): ")
    
    if choice == "1":
        encode_known_faces()
    elif choice == "2":
        image_path = input("Enter the path of the image: ")
        recognize_faces_in_image(image_path)
    elif choice == "3":
        recognize_faces_in_webcam()
    else:
        print("Invalid choice!")
