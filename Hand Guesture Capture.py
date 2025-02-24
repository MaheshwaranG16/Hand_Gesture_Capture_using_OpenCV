import mediapipe as mp
import cv2
import numpy as np
import os

# Configuration
subdir = 'hand_zero'               # Class name for saving images
n_samples_save = 5                # Number of images to save automatically

# Save directory (Modify this path to your desired location)
save_dir = r"C:\Users\mahes\Downloads\CV_Act\Hand Gestures\Captured_Img"
os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists

# Initialize MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Variables
iteration_counter = 1
X, y = [], []
mapping = {
    'hand_closed': 0,
    'hand_three': 1,
    'hand_open': 2,
    'hand_zero': 3,
}

# Initialize Hand Detection
hands = mp_hands.Hands(min_detection_confidence=0.2, static_image_mode=True)
capture = cv2.VideoCapture(0)  # 0 = Laptop Camera, 1 = External Camera

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        print("⚠️ Camera not detected!")
        break

    frame = cv2.flip(frame, 1)  # Flip for better usability
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detected_image = hands.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    one_sample = []

    # Draw hand landmarks
    if detected_image.multi_hand_landmarks:
        for hand_lms in detected_image.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_lms, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=4, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(20, 180, 90), thickness=2, circle_radius=2))

    cv2.imshow('Dataset Maker', image)

    # Automatic Image Saving (First n_samples_save images)
    if iteration_counter <= n_samples_save:
        img_path = os.path.join(save_dir, f'{subdir}_image{iteration_counter}.jpg')
        if cv2.imwrite(img_path, image):
            print(f"✅ Automatically saved: {img_path}")
        else:
            print(f"⚠️ Failed to save image: {img_path}")
        iteration_counter += 1

    # Manual Image Saving (Press "s" to save extra images)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        manual_img_path = os.path.join(save_dir, f'manual_capture_{iteration_counter}.jpg')
        if cv2.imwrite(manual_img_path, image):
            print(f"📸 Manually saved: {manual_img_path}")
        else:
            print(f"⚠️ Failed to save image: {manual_img_path}")

    # Reset Counter (Press "r" to restart automatic saving)
    if cv2.waitKey(1) & 0xFF == ord('r'):
        iteration_counter = 1
        print("🔄 Restarting automatic image capture...")

    # Exit (Press "q" to quit)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
capture.release()
cv2.destroyAllWindows()

# Convert Data to NumPy Arrays
X = np.array(X)
y = np.array(y)
print(f"Dataset Shape: X={X.shape}, y={y.shape}")
