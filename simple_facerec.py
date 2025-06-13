#simple_facerec.py


import face_recognition
import cv2
import os
import glob

class SimpleFacerec:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []

    def load_encoding_images(self, images_path):
        images_path = glob.glob(os.path.join(images_path, "*.*"))

        print(f"🔍 Found {len(images_path)} images for encoding.")

        for img_path in images_path:
            print(f"📷 Loading image: {img_path}")
            img = cv2.imread(img_path)

            if img is None:
                print(f"Failed to load image: {img_path}. Skipping...")
                continue

            try:
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(f"Error converting image to RGB: {img_path}\n{e}")
                continue

            basename = os.path.basename(img_path)
            filename, _ = os.path.splitext(basename)

            encodings = face_recognition.face_encodings(rgb_img)

            if len(encodings) > 0:
                self.known_face_encodings.append(encodings[0])
                self.known_face_names.append(filename)
                print(f"Encoded: {filename}")
            else:
                print(f"No face found in image: {filename}")

    def detect_known_faces(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = "Unknown"

            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            if len(face_distances) > 0:
                best_match_index = face_distances.argmin()
                if matches[best_match_index]:
                    name = self.known_face_names[best_match_index]

            face_names.append(name)

        return face_locations, face_names