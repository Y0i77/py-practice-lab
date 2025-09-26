# encode_faces.py
import os
import face_recognition
import numpy as np
import pickle

DATASET_DIR = "dataset"           # carpeta con subcarpetas por persona
OUTPUT_FILE = "encodings.pkl"     # archivo de salida

known_encodings = []
known_names = []

for person_name in os.listdir(DATASET_DIR):
    person_dir = os.path.join(DATASET_DIR, person_name)
    if not os.path.isdir(person_dir):
        continue
    for filename in os.listdir(person_dir):
        filepath = os.path.join(person_dir, filename)
        try:
            image = face_recognition.load_image_file(filepath)
            # detecta caras y obtiene encoding (128-d vector)
            boxes = face_recognition.face_locations(image, model="hog")  # o model="cnn"
            encs = face_recognition.face_encodings(image, boxes)
            if len(encs) == 0:
                print(f"[WARN] No face found in {filepath}")
                continue
            # si hay varias caras, toma la primera (ajusta si es necesario)
            encoding = encs[0]
            known_encodings.append(encoding)
            known_names.append(person_name)
            print(f"[OK] Procesado {filepath} -> {person_name}")
        except Exception as e:
            print(f"[ERR] {filepath}: {e}")

# Guardar en pickle
with open(OUTPUT_FILE, "wb") as f:
    pickle.dump({"encodings": known_encodings, "names": known_names}, f)

print(f"Guardado {len(known_encodings)} encodings en {OUTPUT_FILE}")
