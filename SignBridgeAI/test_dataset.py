
import os
import cv2
import mediapipe as mp

DATASET_PATH = "new_dataset"

hands_module = mp.solutions.hands

sample_classes = ["A", "B", "J", "Z"]

with hands_module.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.5
) as hands:

    for label in sample_classes:
        folder = os.path.join(DATASET_PATH, label)

        if not os.path.isdir(folder):
            print(f"{label}: folder not found")
            continue

        images = [
            f for f in os.listdir(folder)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ]

        detected = 0
        checked = min(20, len(images))

        for filename in images[:checked]:
            image_path = os.path.join(folder, filename)
            image = cv2.imread(image_path)

            if image is None:
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                detected += 1

        print(f"{label}: detected hands in {detected}/{checked} images")
        