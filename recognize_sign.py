import cv2
import mediapipe as mp
import pickle
import pyttsx3
import time

# -----------------------------
# Load trained AI model
# -----------------------------
with open("models/sign_language_model.pkl", "rb") as file:
    model = pickle.load(file)

# -----------------------------
# Text-to-Speech
# -----------------------------
engine = pyttsx3.init()
engine.setProperty("rate", 150)

# -----------------------------
# MediaPipe
# -----------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# -----------------------------
# Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

# Last spoken sign
last_sign = None
last_speech_time = 0

# Wait before speaking same sign again
speech_cooldown = 2

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

        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hand
        results = hands.process(rgb_frame)

        predicted_sign = "No hand detected"

        if results.multi_hand_landmarks:

            hand_landmarks = results.multi_hand_landmarks[0]

            # Draw landmarks
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # Extract 21 landmarks
            features = []

            for landmark in hand_landmarks.landmark:
                features.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

            # Predict sign
            predicted_sign = model.predict([features])[0]

            # Speak the sign
            current_time = time.time()

            if (
                predicted_sign != last_sign
                or current_time - last_speech_time > speech_cooldown
            ):
                engine.say(predicted_sign.replace("_", " "))
                engine.runAndWait()

                last_sign = predicted_sign
                last_speech_time = current_time

        # -----------------------------
        # Display text
        # -----------------------------
        cv2.putText(
            frame,
            f"Sign: {predicted_sign}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3
        )

        cv2.putText(
            frame,
            "Press Q to quit",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "AI Sign Language Recognition",
            frame
        )

        # Quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()