import cv2
import mediapipe as mp
import csv
import os

# -----------------------------
# Settings
# -----------------------------
SIGN = input("Enter the sign you want to collect (A/B/C/D/E): ").upper()

SAMPLES = 200

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

count = 0

with mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:

    while True:
        success, frame = cap.read()

        if not success:
            print("Could not access camera")
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:

            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw hand landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Collect landmark coordinates
            row = [SIGN]

            for landmark in hand_landmarks.landmark:
                row.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            # Save data
            with open("data/sign_landmarks.csv", "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(row)

            count += 1

        # Display information
        cv2.putText(
            frame,
            f"Sign: {SIGN}  Samples: {count}/{SAMPLES}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("Sign Language Dataset Collection", frame)

        # Stop after collecting enough samples
        if count >= SAMPLES:
            break

        # Press Q to stop manually
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

print(f"\nCollected {count} samples for sign {SIGN}")
print("Dataset saved in data/sign_landmarks.csv")