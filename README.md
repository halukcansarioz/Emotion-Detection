# 🧠 Emotion-Detection

**Artificial Intelligence and Image Based Face Expression Recognition System**

![Emotion Detection Demo](https://github.com/HalukCanSarioz/Emotion-Detection/blob/main/WhatsApp%20Video%202022-05-03%20at%2023.56.44.gif)

Welcome to the **Emotion Detection** repository! This project leverages machine learning and artificial intelligence to analyze and classify human facial expressions using computer vision and deep learning techniques.

## 👨🏼‍💻 About the Developer & Motivation

I have been trying to improve myself since I graduated from Ankara University Computer Engineering. For this, I attend courses from online platforms. I have about 6 months of experience on Ruby on Rails as a Full Stack Developer. I am currently trying to improve myself in Front-end, Back-end, and the fascinating fields of Data Science and AI. The courses I take are on these topics and I develop projects on my own. This repository represents a practical implementation of machine learning concepts I am exploring!

## ✨ Features

*   **Emotion Classification:** Accurately detects and classifies core facial expressions using the FER2013 dataset.
*   **Custom Model Training:** Includes scripts to train, configure, and compile your own emotion detection models.
*   **Computer Vision Integration:** Utilizes OpenCV for real-time or image-based face tracking and bounding box creation.

## 🛠️ Tech Stack

*   **Language:** Python
*   **Machine Learning / Deep Learning:** TensorFlow, Keras
*   **Computer Vision:** OpenCV (`opencv-python`)
*   **Data & Image Processing:** NumPy, Pillow (PIL)
*   **Version Control:** Git & GitHub

## 🚀 Getting Started

Follow these detailed instructions to set up the environment, download the dataset, train the model, and run the emotion detection system on your local machine.

### Prerequisites

Ensure you have Python installed on your system:
*   [Python 3.x](https://www.python.org/downloads/)
*   Git

### Installation & Setup

1.  **Clone the repository and navigate to the directory:**
    
```bash
    git clone [https://github.com/halukcansarioz/Emotion-Detection.git](https://github.com/halukcansarioz/Emotion-Detection.git)
    cd Emotion-Detection
    ```

2.  **Install the required packages:**
    ```bash
    pip install numpy
    pip install opencv-python
    pip install keras
    pip3 install --upgrade tensorflow
    pip install pillow
    ```

3.  **Download the FER2013 Dataset:**
    *   Download the dataset from Kaggle: [FER2013 Dataset](https://www.kaggle.com/msambare/fer2013)
    *   Extract the downloaded files and put them in a folder named `data` under your project directory.

4.  **Train the Emotion Detector:**
    Train the model with all face expression images in the FER2013 Dataset using the following command:
    ```bash
    python TrainEmotionDetector.py
    ```
    *(Additional Training)* I added one more training set to learn. That's why I'm using the same training set in both scripts, but we need to add the data to a folder called `data2` to avoid confusion. To run this secondary training:
    ```bash
    python TrainEmotionDetection.py
    ```
    ⏳ **Note on Training Time:** Training will take several hours depending on your processor. *(On an i7 processor with 16 GB RAM, it took around 4 hours).*

5.  **Configure the Trained Model:**
    After training is complete, you will find the trained model structure and weights stored in your project directory:
    *   `emotion_model.json`
    *   `emotion_model.h5`
    
    Create a new folder named `model` in your project directory, copy these two files, and paste them inside the `model` folder.

6.  **Run the Emotion Detection Test:**
    Execute the test file to see the system in action!
    ```bash
    python TestEmotionDetector.py
    ```

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! 
Feel free to check the [issues page](https://github.com/halukcansarioz/Emotion-Detection/issues) if you want to contribute.

## 📜 License

This project is open-source and available under the MIT License - see the LICENSE file for details.

## 📫 Contact

**Haluk Can Sarıöz**
*   **GitHub:** [@HalukCanSarioz](https://github.com/HalukCanSarioz)
*   **Email:** halukcansarioz19@gmail.com
*   **LinkedIn:** [Haluk Can Sarıöz](https://www.linkedin.com/in/halukcansarioz)

---
*If you find this AI and Computer Vision project useful, please consider giving it a ⭐!*
