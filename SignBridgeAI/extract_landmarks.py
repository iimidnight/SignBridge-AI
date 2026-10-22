
import csv
from pathlib import Path

import cv2
import mediapipe as mp

DATASET_DIR = Path("dataset_raw")
OUTPUT_FILE = Path("data/isl_landmarks.csv")

# Two hands × 21 landmarks × 3 coordinates = 126 features
FEATURE_COUNT = 126


def extract_features(image_path, hands_detector):
    image = cv2.imread(str(image_path))

    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None

    # Sort detected hands from left to right in the image.
    detected_hands = sorted(
        results.multi_hand_landmarks,
        key=lambda hand: sum(p.x for p in hand.landmark)
        / len(hand.landmark)
    )

    features = []

    for hand in detected_hands[:2]:
        for point in hand.landmark:
            features.extend([point.x, point.y, point.z])

    # Pad with zeros when only one hand is detected.
    while len(features) < FEATURE_COUNT:
        features.extend([0.0] * 3)

    return features[:FEATURE_COUNT]


def main():
    if not DATASET_DIR.is_dir():
        print(f"Dataset folder not found: {DATASET_DIR}")
        print("Run this script from your SignBridgeAI project folder.")
        return

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    class_folders = sorted(
        folder for folder in DATASET_DIR.iterdir()
        if folder.is_dir()
    )

    if not class_folders:
        print("No class folders found in dataset_raw.")
        return

    headers = ["label", "image_path"] + [
        f"feature_{i}" for i in range(FEATURE_COUNT)
    ]

    total_images = 0
    total_detected = 0
    total_skipped = 0

    # Static image mode is appropriate for a folder of PNG images.
    with mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        model_complexity=1,
        min_detection_confidence=0.5
    ) as detector:

        with OUTPUT_FILE.open(
            "w", newline="", encoding="utf-8"
        ) as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(headers)

            for class_folder in class_folders:
                images = sorted(
                    path for path in class_folder.rglob("*")
                    if path.is_file()
                    and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
                )

                class_detected = 0
                class_skipped = 0

                for image_path in images:
                    total_images += 1

                    features = extract_features(image_path, detector)

                    if features is None:
                        class_skipped += 1
                        total_skipped += 1
                        continue

                    writer.writerow([
                        class_folder.name,
                        str(image_path),
                        *features
                    ])

                    class_detected += 1
                    total_detected += 1

                print(
                    f"{class_folder.name}: "
                    f"{class_detected}/{len(images)} images processed"
                )

    print("\n--- Processing complete ---")
    print(f"Total images: {total_images}")
    print(f"Images with detected hands: {total_detected}")
    print(f"Skipped images: {total_skipped}")
    print(f"CSV saved to: {OUTPUT_FILE.resolve()}")

    if total_detected == 0:
        print("No hands were detected. Inspect sample images before training.")


if __name__ == "__main__":
    main()
