import os
import cv2
import numpy as np
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("face_recognition_model.keras")

# Load class names
with open("classnames.txt", "r") as f:
    CLASSES = f.read().splitlines()

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load Test Image
test_image_path = "test_image.jpg"
test_img = cv2.imread(test_image_path)
gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

# Detect Faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

for (x, y, w, h) in faces:
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (100, 100))
    face = np.repeat(face[..., np.newaxis], 3, axis=-1)  # Convert to 3-channel
    face = face / 255.0
    face = np.reshape(face, (1, 100, 100, 3))

    # Predict
    predictions = model.predict(face, verbose=0)
    label_idx = np.argmax(predictions)
    confidence = np.max(predictions) * 100

    label = CLASSES[label_idx] if confidence > 50 else "Unknown"

    # Draw bounding box & text
    text = f"{label}: {confidence:.2f}%"
    cv2.rectangle(test_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(test_img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

# Show image
cv2.imshow("Face Recognition Result", test_img)
cv2.waitKey(2000)  # Closes after 2 seconds
cv2.destroyAllWindows()
