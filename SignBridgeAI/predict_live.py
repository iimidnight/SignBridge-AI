
import cv2
import joblib
import numpy as np
import mediapipe as mp

MODEL_PATH = "models/isl_sign_model_v2.pkl"

model = joblib.load(MODEL_PATH)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def extract_features(results):
    detected_hands = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            points = []
            for landmark in hand_landmarks.landmark:
                points.extend([landmark.x, landmark.y, landmark.z])
            detected_hands.append((np.mean(
                [lm.x for lm in hand_landmarks.landmark]
            ), points))

    # Keep the same left-to-right order as training.
    detected_hands.sort(key=lambda item: item[0])

    features = []
    for _, points in detected_hands[:2]:
        features.extend(points)

    while len(features) < 126:
        features.extend([0.0] * 3)

    return np.array(features[:126]).reshape(1, -1)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam. Check camera permissions.")
    hands.close()
    raise SystemExit

print("SignBridge AI running. Press Q to quit.")

try:
    while True:
        success, frame = cap.read()
        if not success:
            print("Could not read from webcam.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        label = "Show a hand"
        confidence = 0.0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

            features = extract_features(results)
            probabilities = model.predict_proba(features)[0]
            best_index = int(np.argmax(probabilities))
            confidence = float(probabilities[best_index])
            label = model.classes_[best_index]

            probabilities = model.predict_proba(features)[0]
            best_index = int(np.argmax(probabilities))
            confidence = float(probabilities[best_index])
            label = model.classes_[best_index]

        cv2.putText(
            frame,
            f"{label} ({confidence:.0%})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        cv2.imshow("SignBridge AI - ISL Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    cap.release()
    cv2.destroyAllWindows()
    hands.close()

