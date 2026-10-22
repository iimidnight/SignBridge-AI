# 🤟 SignBridge AI

### An AI-powered Indian Sign Language Recognition Prototype

SignBridge AI is an open-source project exploring how computer vision and machine learning can support communication by recognising a limited vocabulary of Indian Sign Language (ISL) signs and displaying predicted labels as text.

The project aims to make communication more accessible through a simple, interactive web interface.

> **Project status:** Prototype in development. Recognition accuracy and dataset suitability are still being evaluated. This project is not a complete Indian Sign Language translator.

## ✨ Features

- **Live webcam input:** Capture hand movements through a webcam.
- **Hand landmark detection:** Use MediaPipe to extract hand keypoints.
- **Machine learning classification:** Predict labels from extracted hand landmarks.
- **Sentence builder:** Display predicted labels as editable text.
- **Text editing:** Undo the last word, clear the output, and copy the sentence.
- **Interactive dashboard:** A dark-themed Streamlit interface.

## 🛠️ Tech Stack

- Python
- Streamlit
- OpenCV
- MediaPipe
- NumPy
- scikit-learn
- Joblib
- Streamlit WebRTC

## 📂 Project Structure

```text
SignBridge-AI/
├── app.py
├── app_backup.py
├── collect_data.py
├── collect_isl_data.py
├── extract_landmarks.py
├── predict_live.py
├── train_model.py
├── webcam_test.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
├── dataset_raw/
└── models/
```

The dataset and trained model files may be generated locally and are not necessarily included in this repository.

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/iimidnight/SignBridge-AI.git
cd SignBridge-AI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

Once `requirements.txt` has been created and populated, run:

```bash
pip install -r requirements.txt
```

### 4. Prepare the model

Follow the dataset preparation, landmark extraction, and model training scripts in the repository. A compatible trained model must be available at:

```text
models/isl_sign_model.pkl
```

The current application expects this model file to contain a dictionary with a `model` entry.

### 5. Run the application

```bash
streamlit run app.py
```

Allow webcam access when your browser asks for permission.

## 🧠 How It Works

1. **Input:** The webcam captures a frame.
2. **Landmark extraction:** MediaPipe detects hand landmarks.
3. **Feature preparation:** Landmark coordinates are converted into numerical features.
4. **Classification:** The trained machine learning model predicts a label from its known classes.
5. **Text output:** Stable predictions are added to the sentence builder for the user to review and edit.

## ⚠️ Limitations

- Recognition is limited to the classes represented in the training dataset.
- Predictions can be incorrect, especially when signs look similar or camera conditions vary.
- The current pipeline focuses on static hand landmarks and does not provide full linguistic translation.
- Indian Sign Language includes movement, facial expressions, body posture, grammar, and context that this prototype does not fully capture.
- The application requires a compatible trained model to perform predictions.

## 🗺️ Future Improvements

- Evaluate and improve dataset quality and recognition accuracy.
- Add a verified ISL sign dictionary and learning resources.
- Improve real-time webcam performance and prediction stability.
- Add model evaluation metrics and per-class analysis.
- Improve accessibility, usability, and installation documentation.
- Add tests and clear contribution guidelines.

## 🤝 Contributing

Contributions, suggestions, and bug reports are welcome. Please describe proposed changes clearly and verify dataset licensing before redistributing data.

## 📜 Dataset and Attribution

Dataset sources, licences, and attribution will be documented after the training dataset has been verified.

## 📄 License

A project licence has not yet been selected. Please check this section for updates before reusing or redistributing the code.

---

**Built as a student open-source project exploring AI-assisted accessible communication.**
