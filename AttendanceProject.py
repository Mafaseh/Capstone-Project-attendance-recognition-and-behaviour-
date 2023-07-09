import tensorflow as tf
from tensorflow import keras
import cv2
import dlib
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt

class EmotionDetector:
    def _init_(self, images_path, emotion_cascade_path):
        self.images_path = images_path
        self.emotion_cascade = cv2.CascadeClassifier(emotion_cascade_path)
        self.class_names = []
        self.encode_list_known = []
        self.load_images()

    def load_images(self):
        my_list = os.listdir(self.images_path)
        for cl in my_list:
            cur_img = cv2.imread(f'{self.images_path}/{cl}')
            cur_img_rgb = cv2.cvtColor(cur_img, cv2.COLOR_BGR2RGB)
            face_encodings = self.get_face_encodings(cur_img_rgb)
            if len(face_encodings) > 0:
                self.encode_list_known.append(face_encodings[0])
                self.class_names.append(os.path.splitext(cl)[0])
        print('Encoding Complete')

    def mark_attendance(self, name, behavior):
        with open('Attendance.csv', 'a') as f:  # Open the file in append mode ('a') to add new entries
            now = datetime.now()
            dt_string = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{behavior},{dt_string}')

    def detect_emotions(self):
        cap = cv2.VideoCapture(0)

        # Initialize empty lists for emotion statistics
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        emotion_counts = [0] * len(emotions)

        # Initialize the figure for pie chart visualization
        fig, ax = plt.subplots()

        while True:
            success, img = cap.read()
            if img is None:
                continue

            img_s = cv2.resize(img, (0, 0), None, 2, 2)
            img_s_gray = cv2.cvtColor(img_s, cv2.COLOR_BGR2GRAY)

            faces_cur_frame = self.emotion_cascade.detectMultiScale(img_s_gray, scaleFactor=1.1, minNeighbors=5,
                                                                    minSize=(30, 30))

            dominant_emotion = None  # Store the dominant emotion

            for (x, y, w, h) in faces_cur_frame:
                x1, y1, x2, y2 = x, y, x + w, y + h
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                roi = img_s[y1:y2, x1:x2]

                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                face_encodings = self.get_face_encodings(roi_rgb)
                if len(face_encodings) > 0:
                    behaviors = self.analyze_behavior(face_encodings[0])
                    dominant_emotion = max(set(behaviors), key=behaviors.count)  # Get the dominant emotion

            if dominant_emotion:
                self.mark_attendance("Your Name", dominant_emotion)  # Pass your name and dominant_emotion parameters
                emotion_counts[emotions.index(dominant_emotion)] += 1

            # Update the pie chart with the current emotion statistics
            ax.clear()
            if sum(emotion_counts) > 0:
                ax.pie(emotion_counts, labels=emotions, autopct='%1.1f%%')
            ax.set_aspect('equal')
            ax.set_title('Emotion Distribution')

            # Show the pie chart in a new window
            plt.pause(0.01)

            cv2.imshow('Webcam', img)
            if cv2.waitKey(1) == 27:  # Press Esc to exit
                break

        cap.release()
        cv2.destroyAllWindows()

    def analyze_behavior(self, face_encoding):
        # Load the pre-trained facial expression recognition model
        model = keras.models.load_model(
            'C:\\Users\\gateway\\Desktop\\Face-Recognition-master\\fer2013_mini_XCEPTION.102-0.66.hdf5', compile=False)
        emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

        # Resize and preprocess the face encoding for the model
        face_encoding = cv2.resize(face_encoding, (64, 64))  # Resize to (64, 64)
        face_encoding = face_encoding / 255.0
        face_encoding = np.expand_dims(face_encoding, axis=0)
        face_encoding = np.expand_dims(face_encoding, axis=-1)

        # Predict the facial expression using the pre-trained model
        predictions = model.predict(face_encoding)
        predicted_emotion = emotions[np.argmax(predictions)]

        return [predicted_emotion]

    def get_face_encodings(self, image):
        face_encodings = []
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        faces = self.emotion_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        for (x, y, w, h) in faces:
            face_encoding = gray[y:y + h, x:x + w]  # Extract the face region as a numpy array
            face_encodings.append(face_encoding)
        return face_encodings

    def save_average_emotions_to_csv(self, average_emotion_counts):
        with open('AverageEmotions.csv', 'w') as f:
            f.write('Emotion,Percentage\n')
            emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
            for emotion, percentage in zip(emotions, average_emotion_counts):
                f.write(f'{emotion},{percentage * 100}\n')


# Specify the paths to the images folder and emotion cascade XML file
images_path = r'C:\Users\gateway\Desktop\Face-Recognition-master\Images'
emotion_cascade_path = r'C:\Users\gateway\Desktop\Face-Recognition-master\haarcascade_frontalface_default.xml'

# Create an instance of the EmotionDetector class
emotion_detector = EmotionDetector(images_path=images_path, emotion_cascade_path=emotion_cascade_path)

# Start detecting emotions and marking attendance
emotion_detector.detect_emotions()