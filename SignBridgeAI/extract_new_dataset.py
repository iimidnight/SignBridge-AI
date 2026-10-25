
import os
import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

DATASET_PATH = "new_dataset"
OUTPUT_PATH = "data/isl_landmarks_new.csv"

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

os.makedirs("data", exist_ok=True)

mp_hands = mp.solutions.hands


def extract_features(image, hands_detector):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands_detector.process(rgb)

    # Two slots, 21 landmarks per hand, x/y/z per landmark.
    hand_slots = [
        np.zeros(63, dtype=np.float32),
        np.zeros(63, dtype=np.float32)
    ]

    if result.multi_hand_landmarks:
        detected_hands = []

        for hand_landmarks in result.multi_hand_landmarks:
            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                dtype=np.float32
            )
            detected_hands.append(coords.flatten())

        # Keep hand ordering consistent within each image:
        # leftmost hand first, then rightmost.
        detected_hands.sort(key=lambda values: np.mean(values[0::3]))

        for i, features in enumerate(detected_hands[:2]):
            hand_slots[i] = features

    return np.concatenate(hand_slots)


def main():
    rows = []
    total = 0
    failed = 0

    classes = sorted(
        folder for folder in os.listdir(DATASET_PATH)
        if os.path.isdir(os.path.join(DATASET_PATH, folder))
    )

    print("Classes found:", classes)

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        model_complexity=0,
        min_detection_confidence=0.5
    ) as detector:

        for label in classes:
            folder = os.path.join(DATASET_PATH, label)
            files = [
                f for f in os.listdir(folder)
                if f.lower().endswith(IMAGE_EXTENSIONS)
            ]

            print(f"\nProcessing {label}: {len(files)} images")

            for filename in files:
                path = os.path.join(folder, filename)
                image = cv2.imread(path)

                if image is None:
                    failed += 1
                    continue

                features = extract_features(image, detector)

# Skip images where MediaPipe detected no hand.
                if not np.any(features):
                  failed += 1
                  continue

                rows.append({
                **{
                f"f{i}": float(value)
                for i, value in enumerate(features)
                },
               "label": label
                })

                total += 1

                if total % 500 == 0:
                    print(f"Processed {total} images...")

    if not rows:
        print("No images were processed. Check DATASET_PATH.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\nExtraction complete!")
    print("Images processed:", total)
    print("Images unreadable:", failed)
    print("CSV saved to:", OUTPUT_PATH)
    print("CSV shape:", df.shape)
    print("\nSamples per class:")
    print(df["label"].value_counts().sort_index())


if __name__ == "__main__":
    main()
