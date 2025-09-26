# recognize_camera.py
import cv2
import face_recognition
import pickle
import time
import numpy as np

ENCODINGS_FILE = "encodings.pkl"
TOLERANCE = 0.5           # umbral para matches (0.4 más estricto, 0.6 más laxo)
MODEL = "hog"             # "hog" (CPU) o "cnn" (GPU si está disponible)

# Cargar encodings
with open(ENCODINGS_FILE, "rb") as f:
    data = pickle.load(f)
known_encodings = data["encodings"]
known_names = data["names"]

# Webcam
cap = cv2.VideoCapture(0)   # 0 o el índice de tu cámara

process_every_n_frames = 2
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Reducir tamaño para velocidad
    small_frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)
    rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    if frame_count % process_every_n_frames == 0:
        face_locations = face_recognition.face_locations(rgb_small, model=MODEL)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

        face_names = []
        for encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, encoding, tolerance=TOLERANCE)
            name = "Unknown"

            # usamos la distancia para elegir el mejor match
            face_distances = face_recognition.face_distance(known_encodings, encoding)
            if len(face_distances) > 0:
                best_idx = np.argmin(face_distances)
                if matches[best_idx]:
                    name = known_names[best_idx]
            face_names.append(name)

    frame_count += 1

    # Mostrar resultados (escalamos coordenadas)
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # volver a tamaño original
        top *= 2; right *= 2; bottom *= 2; left *= 2
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.rectangle(frame, (left, bottom-20), (right, bottom), (0,255,0), cv2.FILLED)
        cv2.putText(frame, name, (left+2, bottom-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)

    cv2.imshow("Reconocimiento CIPA", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
