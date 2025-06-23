# recognize_and_log.py

import cv2
from deepface import DeepFace
import pandas as pd
import os
from datetime import datetime


def log_attendance(user_id):
    log_path = 'logs/attendance.csv'
    os.makedirs("logs", exist_ok=True)

    now = datetime.now()
    time_str = now.strftime("%Y-%m-%d %H:%M:%S")

    entry = {"user_id": user_id, "timestamp": time_str}

    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    else:
        df = pd.DataFrame([entry])

    df.to_csv(log_path, index=False)
    print(f"[LOGGED] {user_id} at {time_str}")


def recognize_from_webcam():
    print("[INFO] Loading webcam...")
    cam = cv2.VideoCapture(0)

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        try:
            results = DeepFace.find(
                img_path=frame, db_path="dataset/", enforce_detection=False, silent=True)

            if results[0].shape[0] > 0:
                identity_path = results[0].iloc[0]["identity"]
                user_id = identity_path.split("/")[-2]
                log_attendance(user_id)

                cv2.putText(frame, f"{user_id}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Unknown", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        except Exception as e:
            print("[WARN] Face not detected")

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


# Example usage
# recognize_from_webcam()
if __name__ == "__main__":
    recognize_from_webcam()
