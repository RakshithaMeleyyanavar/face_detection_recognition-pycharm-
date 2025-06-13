import cv2
from simple_facerec import SimpleFacerec

# Load face encodings
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Start webcam
cap = cv2.VideoCapture(0)
#Resolution should is increased to detect the farther faces
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

while True:
    ret, frame = cap.read()
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=20)

    # Detect faces
    face_locations, face_names = sfr.detect_known_faces(frame)

    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = [v * 2 for v in face_loc]  # Scale back up: X2-shrinking the frame to 0.5x, OR X4-shrinking the frame to 0.25x
        cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 200), 4)

    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # Press 'Esc' to exit
        break

cap.release()
cv2.destroyAllWindows()