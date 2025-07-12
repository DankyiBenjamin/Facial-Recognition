# register.py

import cv2
import os

# -----------------------------
# Function to register a new user's face
# -----------------------------
def register_user(user_id):
    # Create directory path for storing the user's image
    save_path = os.path.join("dataset", user_id)
    os.makedirs(save_path, exist_ok=True)  # Create directory if it doesn't exist

    # Open the default webcam
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("[ERROR] Could not open webcam.")
        return

    print(f"[INFO] Registering {user_id}... Press 's' to save or 'q' to cancel.")

    while True:
        ret, frame = cam.read()
        if not ret:
            break  # Exit loop if frame capture fails

        # Show the webcam feed in a window
        cv2.imshow("Register Face", frame)

        # Wait for key input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            # Save the current frame to disk when 's' is pressed
            cv2.imwrite(os.path.join(save_path, "face.jpg"), frame)
            print(f"[SUCCESS] Face saved to {save_path}")
            break  # Exit after saving

        elif key == ord('q'):
            # Cancel registration if 'q' is pressed
            print("[INFO] Registration cancelled.")
            break

    # Release camera and close window
    cam.release()
    cv2.destroyAllWindows()
