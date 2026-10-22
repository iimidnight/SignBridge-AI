
import cv2
import mediapipe as mp
import csv
import os

# Save path for manually collected landmark samples
DATA_FILE = os.path.join("data", "isl_landmarks.csv")
os.makedirs("data", exist_ok=True)

# Temporary labels only
LABELS = {
    ord("1"): "SIGN_1",
    ord("2"): "SIGN_2"
}

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

camera = None
hands = None
saved_count = 0
current_label = None

try:
    print("Starting SignBridge collector...")

    # Initialize MediaPipe
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    )

    # Open webcam
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        raise RuntimeError(
            "Camera could not be opened. Close other camera apps "
            "and check Windows camera permissions."
        )

    print("Camera opened successfully.")
    print("Controls: 1 = SIGN_1, 2 = SIGN_2, SPACE = save, Q = quit")

    # Append samples without deleting existing data
    with open(DATA_FILE, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        if file.tell() == 0:
            writer.writerow(
                ["label"] + [f"f{i}" for i in range(63)]
            )

        while True:
            success, frame = camera.read()

            if not success or frame is None:
                print("Could not read a webcam frame.")
                break

            # Mirror the camera image
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect hand landmarks
            result = hands.process(rgb)
            features = None

            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]

                mp_draw.draw_landmarks(
                    frame,
                    hand,
                    mp_hands.HAND_CONNECTIONS
                )

                features = []
                for point in hand.landmark:
                    features.extend([point.x, point.y, point.z])

                message = (
                    f"Selected: {current_label}"
                    if current_label
                    else "Press 1 or 2 to select a label"
                )

                cv2.putText(
                    frame, message, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 255, 0), 2
                )

                cv2.putText(
                    frame, "SPACE: save sample",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 0), 2
                )
            else:
                cv2.putText(
                    frame, "Show one hand to the camera",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 0, 255), 2
                )

            cv2.putText(
                frame,
                "1: SIGN_1 | 2: SIGN_2 | Q: quit",
                (10, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2
            )

            cv2.imshow("SignBridge - Collect Samples", frame)

            # Process keyboard input from the OpenCV window
            key = cv2.waitKey(1) & 0xFF

            if key in LABELS:
                current_label = LABELS[key]
                print("Selected label:", current_label)

            elif key == ord(" ") and features is not None:
                if current_label is None:
                    print("Select a label first by pressing 1 or 2.")
                else:
                    writer.writerow([current_label] + features)
                    file.flush()
                    saved_count += 1
                    print(f"Saved sample {saved_count}: {current_label}")

            elif key == ord("q"):
                print("Quit requested.")
                break

except KeyboardInterrupt:
    print("\nCollector stopped with Ctrl+C.")

except Exception as error:
    print(f"Collector error: {error}")

finally:
    if camera is not None:
        camera.release()

    if hands is not None:
        hands.close()

    cv2.destroyAllWindows()

print(f"Samples saved this session: {saved_count}")
print(f"Data file: {DATA_FILE}")

