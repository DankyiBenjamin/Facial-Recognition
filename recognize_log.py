# recognizer.py

import cv2
from deepface import DeepFace
import pandas as pd
import os
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def log_attendance(user_id, log_path='logs/attendance.csv'):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = {"user_id": user_id, "timestamp": timestamp}

    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    else:
        df = pd.DataFrame([entry])
    df.to_csv(log_path, index=False)
    print(f"[LOGGED] {user_id} at {timestamp}")


def face_detected(frame):
    """
    Uses OpenCV's Haar Cascade to check if a face exists in the frame.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return len(faces) > 0


def get_best_match(frame, db_path, threshold=0.6):
    """
    Gets the best matching user based on cosine similarity.
    """
    try:
        # Represent the frame
        target = DeepFace.represent(
            frame, model_name="Facenet", enforce_detection=True)[0]["embedding"]
    except Exception as e:
        print(f"[WARN] Face encoding failed: {e}")
        return None

    best_match = None
    best_score = -1

    # Loop through each registered user image in dataset/
    for user_id in os.listdir(db_path):
        user_dir = os.path.join(db_path, user_id)
        face_path = os.path.join(user_dir, "face.jpg")

        if not os.path.exists(face_path):
            continue

        try:
            source = DeepFace.represent(
                img_path=face_path, model_name="Facenet", enforce_detection=True)[0]["embedding"]
            sim = cosine_similarity([target], [source])[0][0]

            if sim > best_score and sim >= threshold:
                best_score = sim
                best_match = user_id
        except Exception as e:
            print(f"[WARN] Failed to compare with {user_id}: {e}")
            continue

    return best_match


def recognize_and_log(db_path='dataset/'):
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("[ERROR] Webcam not accessible.")
        return

    print("[INFO] Looking for known faces... Press 'q' to quit.")

    matched_users = set()

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        if not face_detected(frame):
            cv2.putText(frame, "No face detected", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        user_id = get_best_match(frame, db_path, threshold=0.6)

        if user_id:
            if user_id not in matched_users:
                log_attendance(user_id)
                matched_users.add(user_id)
                print(f"[INFO] Match found: {user_id}")

            cv2.putText(frame, f"{user_id}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Unknown", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        cv2.imshow("Face Recognition", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Quit manually.")
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    recognize_and_log()
    print("[INFO] Attendance logging complete.")
