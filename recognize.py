import cv2
import numpy as np

# Function to draw boundaries and label recognized faces
def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, clf):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)
    
    for (x, y, w, h) in features:
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        id, pred = clf.predict(gray_image[y:y+h, x:x+w])
        
        confidence = int(100 * (1 - pred / 300))
        
        if confidence > 77:
            if id == 1:
                label = 'Criminal'
                color = (255, 0, 0)  # Red for Criminal
            elif id == 2:
                label = 'Police'
                color = (0, 0, 255)  # Blue for Police
            else:
                label = 'Civilian'
                color = (0, 255, 0)  # Green for Civilian
        else:
            label = 'Civilian'
            color = (255, 255, 0)  # Yellow for unknown
        
        cv2.putText(img, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)

    return img

# Function to recognize faces in the video
def recognize(img, clf, faceCascade):
    return draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 255), clf)

# Main workflow for recognition
# Load the trained classifier
faceCascade = cv2.CascadeClassifier("C:/Users/lakshay pandey/Desktop/DSS/haarcascade_frontalface_default.xml")
clf = cv2.face.LBPHFaceRecognizer_create()
clf.read('classifier.xml')

# Start video capture
video_capture = cv2.VideoCapture(0)

while True:
    ret, img = video_capture.read()
    if not ret:
        break
    img = recognize(img, clf, faceCascade)
    cv2.imshow('Face Detection', img)
    
    if cv2.waitKey(1) == 13:  # Press Enter to exit
        break

video_capture.release()
cv2.destroyAllWindows()
