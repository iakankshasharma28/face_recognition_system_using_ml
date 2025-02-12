import os
import cv2

# Define dataset folder
DATASET_DIR = "dataset"
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

# Capture images for a person
person_name = input("Enter the person's name: ")
person_folder = os.path.join(DATASET_DIR, person_name)

if not os.path.exists(person_folder):
    os.makedirs(person_folder)

# Initialize webcam
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

count = 0
while count < 100:  # Capture 100 images per person
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100))
        cv2.imwrite(f"{person_folder}/{count}.jpg", face)
        count += 1

        # Show image with bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Face Capture", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
print(f"✅ Captured 100 images for {person_name}!")
