# 🎭 Emotion Detection
### (Artificial Intelligence and Image-Based Face Expression Recognition System)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](#)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)](#)
[![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)](#)

This project is a **real‑time facial expression recognition system** powered by deep learning (CNN). It was developed as a capstone project for the **Department of Computer Engineering at Ankara University**.

## 📚 Table of Contents
- [About the Project](#about-the-project)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation & Usage](#installation--usage)
- [Project Structure](#project-structure)
- [Dataset Information](#dataset-information)
- [Development Process](#development-process)
- [Contributing](#contributing)
- [Contact](#contact)
- [License](#license)

---

## About the Project
This work combines image processing and artificial intelligence techniques to detect an individual's emotional state in real time. The model, trained on the **FER‑2013** dataset, analyzes facial data captured from a camera feed and classifies it into **7 distinct emotion categories**.

- **Developers:** Haluk Can SARIÖZ & Mesut ÖZLAHLAN
- **Advisor:** Research Assistant İrem ÜLKÜ
- **Institution:** Ankara University, Faculty of Engineering, Department of Computer Engineering

---

## Features
- **Real‑Time Detection** — Instant analysis of live video streams with low latency.
- **7 Basic Emotions** — Angry, Disgusted, Fearful, Happy, Neutral, Sad, and Surprised.
- **Visual Feedback** — Bounding boxes around detected faces with live emotion labels.
- **Intelligent Architecture** — Optimized Convolutional Neural Network (CNN) design.

---

## Technologies Used
- **OpenCV** — Camera access and Haar Cascade face detection.
- **Keras & TensorFlow** — Deep learning model training and inference.
- **NumPy & Pandas** — Data management and matrix operations.
- **Matplotlib** — Training process analysis and charting.

---

## Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/halukcansarioz/Emotion-Detection.git
```

### 2. Navigate to the Project Directory
```bash
cd Emotion-Detection
```

### 3. Install Dependencies
The required libraries are listed below. Install them manually:
```bash
pip install opencv-python tensorflow keras numpy pandas matplotlib
```

### 4. Launch the Application
To run the real‑time emotion detection on your webcam:
```bash
python Video.py
```
To test the trained model on sample images:
```bash
python TestEmotionDetector.py
```

> ⚠️ **Note:** Ensure your camera is not in use by another application before running `Video.py`.

---

## Project Structure
```text
Emotion-Detection/
├── .gitattributes                  # Git configuration file
├── TestEmotionDetector.py          # Model testing script (images)
├── TrainEmotionDetection.py        # Model training script
├── TrainEmotionDetector.py         # Model training script (alternate)
├── Video.py                        # Real‑time webcam emotion detection
├── haarcascades/                   # Haar Cascade XML files for face detection
├── model/                          # Trained model weights (.h5 files)
├── Sample_videos/                  # Sample video files for testing
└── README.md                       # Project documentation
```

---

## Dataset Information
The **FER‑2013** dataset was used for training.
- **Content:** 48×48 pixel grayscale face images.
- **Scope:** Approximately 35,000 samples across 7 emotion classes.

---

## Development Process

### 1. Fork the Repository
Start by forking the project to your own GitHub account.

### 2. Create a New Branch
```bash
git checkout -b feature/model-improvement
```

### 3. Push Your Code
```bash
git push origin feature/model-improvement
```

---

## Contributing
1. **Fork** this repository.
2. Create a **Branch** (`git checkout -b feature/NewFeature`).
3. Make your changes and **Commit** (`git commit -m 'Add: New feature'`).
4. **Push** your code (`git push origin feature/NewFeature`).
5. Open a **Pull Request**.

---

<a name="contact"></a>
## Contact
**Haluk Can Sarıöz**
- GitHub: [@halukcansarioz](https://github.com/halukcansarioz)
- Email: [halukcansarioz19@gmail.com](mailto:halukcansarioz19@gmail.com)
- LinkedIn: [Haluk Can Sarıöz](https://www.linkedin.com/in/halukcansarioz)

**Mesut ÖZLAHLAN**
- GitHub: [@mesutozlahlan](https://github.com/mesutozlahlan)

**Project Link:** [https://github.com/halukcansarioz/Emotion-Detection](https://github.com/halukcansarioz/Emotion-Detection)

---

## License
This project is licensed under the [MIT License](LICENSE).
