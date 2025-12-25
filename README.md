**🧠 Gender Detection Using UTKFace Dataset**
(Simple Image Detection + Webcam Detection)

This project implements a Gender Detection System using a Convolutional Neural Network (CNN) trained on the UTKFace dataset.
It supports two modes of prediction:

  Simple Gender Detection – predict gender from an image
  Webcam Gender Detection – predict gender using live webcam input

The dataset is downloaded using kagglehub, so no Kaggle API key or authentication is required.

**📌 Features**

✅ Dataset download using kagglehub
✅ Automatic label extraction from UTKFace filenames
✅ Train/Test split (80/20)
✅ CNN-based gender classification
✅ Simple image-based gender detection
✅ Live webcam gender detection
✅ Google Colab compatible

**📂 Dataset**

UTKFace Dataset

Filename format:

age_gender_race_date.jpg
gender = 0 → Male
gender = 1 → Female

Dataset source:

jangedoo/utkface-new (via kagglehub)

**🛠️ Tech Stack**

Python
PyTorch
torchvision
OpenCV
kagglehub
Google Colab

**🚀 Project Workflow**

Download UTKFace dataset
Extract images
Split dataset into train/test
Train CNN model
Perform gender prediction using:
Simple image input
Webcam input



**🧠 Model Architecture**

Convolutional Layers (3)
ReLU Activation
MaxPooling
Fully Connected Layers
Output Classes: Male / Female

**🏋️ Training the Model**

Optimizer: Adam
Loss Function: CrossEntropyLoss
Epochs: 5
Training code is available in:

gender_detection_train.ipynb

**🖼️ Simple Gender Detection (Image Input)**

This mode predicts gender from a single image.

📁 File:

simple_gender_detection.ipynb
Workflow:
Load trained model
Load an image
Resize to 128×128
Predict gender
Example Output:
Predicted Gender: Male

**📷 Webcam Gender Detection (Live)**

This mode uses the webcam to capture a face image and predict gender in real-time.

📁 File:

gender_detection_webcam.ipynb
Workflow:
Capture image via browser webcam
Preprocess image
Feed into trained CNN
Display predicted gender
Example Output:
Detected Gender: Female

**✅ Results**

CNN successfully classifies gender from facial images
Works on both static images and live webcam input
Lightweight and beginner-friendly implementation

**⚠️ Notes**

Webcam access requires browser permission
Good lighting improves accuracy
Model performance depends on image quality

**🔮 Future Enhancements**

Face detection before classification
Transfer learning (ResNet / MobileNet)
Desktop real-time webcam app
Android deployment

**👤 Author**
Mehak Zahra
