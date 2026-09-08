import cv2
import mediapipe as mp
import csv
import os

# -----------------------------
# Settings
# -----------------------------
SIGNS = [
    "Hi",
    "Good",
    "OK",
    "I_Love_You",
    "Power",
    "Gratitude",
    "Victory",
    "Point_Right",
    "Stop",
    "Heart"
]

SAMPLES = 200

# Create data folder
os.makedirs("data", exist_ok=True)

# MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Ask which sign to collect
print("\nAvailable signs:")
for i, sign in enumerate(SIGNS, 1):
    print(f"{i}. {sign}")

choice = int(input("\nEnter the number of the sign: "))

if choice < 1 or choice > len(SIGNS):
    print("Invalid choice!")
    exit()

SIGN = SIGNS[choice - 1]

print(f"\nCollecting samples for: {SIGN}")
print("Make the sign clearly in front of the camera.")
print("Move your hand slightly while maintaining the sign.")
print("Press Q to stop early.\n")

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

        # Convert BGR → RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hand
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:

            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Store label
            row = [SIGN]

            # Store 21 landmark coordinates
            for landmark in hand_landmarks.landmark:
                row.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            # Append to CSV
            with open(
                "data/custom_signs.csv",
                "a",
                newline=""
            ) as file:

                writer = csv.writer(file)
                writer.writerow(row)

            count += 1

        # Display progress
        cv2.putText(
            frame,
            f"Sign: {SIGN}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Samples: {count}/{SAMPLES}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow(
            "Custom Sign Dataset Collection",
            frame
        )

        # Stop after 200 samples
        if count >= SAMPLES:
            break

        # Press Q to stop
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()

print("\n--------------------------------")
print(f"Collected {count} samples for: {SIGN}")
print("Saved to: data/custom_signs.csv")
print("--------------------------------")