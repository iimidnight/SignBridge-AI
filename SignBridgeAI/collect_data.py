
import cv2
import mediapipe as mp
import csv
import os

DATA_FILE = "data/landmarks.csv"

# These are temporary labels for testing the collection pipeline.
# Replace them with verified ISL signs before building the final dataset.
LABELS = ["TEST_A", "TEST_B"]

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

os.makedirs("data", exist_ok=True)

file_exists = os.path.exists(DATA_FILE)

with open(DATA_FILE, "a", newline="") as file:
    writer = csv.writer(file)

    if not file_exists or os.path.getsize(DATA_FILE) == 0:
        writer.writerow(
            ["label"] +
            [f"x{i}" for i in range(21)] +
            [f"y{i}" for i in range(21)] +
            [f"z{i}" for i in range(21)]
        )

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6
    ) as hands:

        camera = cv2.VideoCapture(0)

        if not camera.isOpened():
            raise RuntimeError("Could not open webcam.")

        print("Press A to collect TEST_A samples.")
        print("Press B to collect TEST_B samples.")
        print("Press Q to quit.")

        current_label = None
        collected = {"TEST_A": 0, "TEST_B": 0}

        try:
            while True:
                success, frame = camera.read()
                if not success:
                    print("Could not read webcam.")
                    break

                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb)

                if result.multi_hand_landmarks:
                    hand = result.multi_hand_landmarks[0]
                    mp_draw.draw_landmarks(
                        frame, hand, mp_hands.HAND_CONNECTIONS
                    )

                    if current_label:
                        points = hand.landmark
                        xs = [p.x for p in points]
                        ys = [p.y for p in points]
                        zs = [p.z for p in points]

                        writer.writerow(
                            [current_label] + xs + ys + zs
                        )
                        file.flush()
                        collected[current_label] += 1

                status = current_label or "Paused - press A or B"
                cv2.putText(
                    frame, f"Label: {status}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2
                )
                cv2.putText(
                    frame,
                    f"A: {collected['TEST_A']}  B: {collected['TEST_B']}  Q: quit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2
                )

                cv2.imshow("SignBridge - Data Collection", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("a"):
                    current_label = "TEST_A"
                elif key == ord("b"):
                    current_label = "TEST_B"
                elif key == ord("q"):
                    break

        finally:
            camera.release()
            cv2.destroyAllWindows()

print(f"Saved samples to {DATA_FILE}")
