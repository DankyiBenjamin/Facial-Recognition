# recognizer.py

import cv2
from deepface import DeepFace
import pandas as pd
import os
from datetime import datetime

# -----------------------------
# Function to log attendance
# -----------------------------
def log_attendance(user_id, log_path='logs/attendance.csv'):
    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    # Generate current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Create attendance entry
    entry = {"user_id": user_id, "timestamp": timestamp}

    # Load existing CSV or create new one
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        df = pd.concat([df, pd.DataFrame([entry])], ignore_index=True)
    else:
        df = pd.DataFrame([entry])

    # Save updated attendance log
    df.to_csv(log_path, index=False)
    print(f"[LOGGED] {user_id} at {timestamp}")

# -----------------------------
# Function to perform recognition
# -----------------------------
def recognize_and_log():
    # Initialize webcam
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("[ERROR] Webcam not accessible.")
        return

    recognized = set()  # Track already recognized users for this session

    print("[INFO] Press 'q' to quit manually or wait for match.")

    while True:
        ret, frame = cam.read()
        if not ret:
            break  # Exit loop if webcam read fails

        try:
            # Try to find a matching face in the dataset using DeepFace
            results = DeepFace.find(
                # Use the current frame as input
                img_path=frame,
                # specify where the faces are stored
                db_path="dataset/",
                enforce_detection=False,  # Don't crash if no face is detected
                silent=True
            )

            # If a match is found
            if not results[0].empty:
                # Extract user_id from matched file path
                identity_path = results[0].iloc[0]["identity"]
                user_id = os.path.basename(os.path.dirname(identity_path))

                # Log attendance only once per session
                if user_id not in recognized:
                    recognized.add(user_id)
                    log_attendance(user_id)

                    # Draw user_id label on frame
                    cv2.putText(frame, f"{user_id}", (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

                    print(f"[INFO] Recognized {user_id}. Exiting...")

            else:
                # No match found – mark as unknown
                cv2.putText(frame, "Unknown", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)

        except Exception as e:
            print(f"[WARN] Face detection failed: {e}")

        # Show webcam frame
        cv2.imshow("Face Recognition", frame)

        # Manual exit with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Manual quit.")
            break

    # Release resources
    cam.release()
    cv2.destroyAllWindows()
