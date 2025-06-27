# register_faces.py

import cv2
import os
import sys
import numpy as np
from deepface import DeepFace


def register_user(user_id):
    image_dir = os.path.join("dataset", user_id)
    embedding_dir = "embeddings"
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(embedding_dir, exist_ok=True)

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("[ERROR] Webcam not accessible.")
        return

    print(f"[INFO] Registering {user_id}. Press 's' to save, 'q' to quit.")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        cv2.imshow("Register Face", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            img_path = os.path.join(image_dir, "face.jpg")
            cv2.imwrite(img_path, frame)
            print(f"[SAVED] Image saved at {img_path}")

            try:
                # Extract embedding using DeepFace
                result = DeepFace.represent(
                    img_path=img_path, model_name="Facenet", enforce_detection=True)
                embedding = result[0]["embedding"]

                # Save embedding
                np.save(os.path.join(embedding_dir,
                        f"{user_id}.npy"), embedding)
                print(f"[ENCODED] Embedding saved for {user_id}")
            except Exception as e:
                print(f"[ERROR] Failed to create embedding: {e}")
            break

        elif key == ord('q'):
            print("[INFO] Registration canceled.")
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        register_user(sys.argv[1])
    else:
        print("Usage: python register_faces.py <user_id>")
