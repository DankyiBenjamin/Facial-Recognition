# register_face.py

import cv2
import os


def register_user(user_id):
    save_path = f"dataset/{user_id}"
    os.makedirs(save_path, exist_ok=True)

    cam = cv2.VideoCapture(0)
    print(f"[INFO] Capturing image for {user_id}. Press 's' to save.")

    while True:
        ret, frame = cam.read()
        if not ret:
            break

        cv2.imshow("Register Face", frame)

        key = cv2.waitKey(1)
        if key == ord('s'):
            img_path = f"{save_path}/face.jpg"
            cv2.imwrite(img_path, frame)
            print(f"[INFO] Saved to {img_path}")
            break
        elif key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

# Example usage
# register_user("john_doe")


if __name__ == "__main__":
    user_id = input("Enter user ID: ")
    register_user(user_id)
