# Emotion-Detection
Artificial Intelligence and Image Based Face Expression Recognition System

![ezgif-7-d44384a1f4e7](https://github.com/HalukCanSarioz/Emotion-Detection/blob/main/WhatsApp%20Video%202022-05-03%20at%2023.56.44.gif)

### Packages need to be installed
- pip install numpy
- pip install opencv-python
- pip install keras
- pip3 install --upgrade tensorflow
- pip install pillow

### download FER2013 dataset
- from below link and put in data folder under your project directory
- https://www.kaggle.com/msambare/fer2013

### Train Emotion detector
- with all face expression images in the FER2013 Dataset
- command --> python TrainEmotionDetector.py
- I added one more training set to learn. That's why I'm using the same training set in both, but we need to add it to a folder called data2 to avoid confusion.
- command --> python TrainEmotionDetection.py

It will take several hours depends on your processor. (On i7 processor with 16 GB RAM it took me around 4 hours)
after Training , you will find the trained model structure and weights are stored in your project directory.
emotion_model.json
emotion_model.h5

copy these two files create model folder in your project directory and paste it.

### run your emotion detection test file
python TestEmotionDetector.py
