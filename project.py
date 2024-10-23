import cv2
import numpy as np
import os
from PIL import Image 

# Function to create dataset folder for the user based on their ID
def create_folder_for_id(id):
    folder_path = f"C:/Users/lakshay pandey/Desktop/DSS/data/user_{id}/"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  # Create a new folder for each user based on ID
    return folder_path

# Function to collect face data
def generate_dataset(id):
    face_classifier = cv2.CascadeClassifier("C:/Users/lakshay pandey/Desktop/DSS/haarcascade_frontalface_default.xml")
    
    def face_crop(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y+h, x:x+w]
        return cropped_face
    
    cap = cv2.VideoCapture(0)
    img_id = 0
    
    folder_path = create_folder_for_id(id)  # Create a folder for this ID

    print(f"Collecting samples for ID: {id}...")
    while img_id < 100:  # Collect 100 images
        ret, frame = cap.read()
        if face_crop(frame) is not None:
            img_id += 1
            face = cv2.resize(face_crop(frame), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path = os.path.join(folder_path, f"user.{id}.{img_id}.jpg")  # Store images in the specific folder for this ID
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Cropped Face:", face)
            if cv2.waitKey(1) == 13:  # Press Enter to stop
                break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Collecting samples is completed...")

# Main workflow
if __name__ == "__main__":
    id = int(input("Enter the ID for the person (e.g., 1 for Criminal, 2 for Police): "))  # User can enter ID
    generate_dataset(id)
