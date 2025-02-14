# 🥦🍎 Fruit & Vegetable Recognition

This repository contains a machine learning model designed to recognize fruits and vegetables using deep learning and image processing techniques. The project is built using TensorFlow and OpenCV for classification.

## 📌 Features
- Identifies various fruits and vegetables from images.
- Uses a trained neural network model for accurate classification.
- Simple setup and execution with Python.

---

## 🚀 Installation & Setup

### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/your-username/fruit-vegetable-recognition.git
cd fruit-vegetable-recognition
```

### 2️⃣ Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate  # On Windows
```

### 3️⃣ Install Dependencies
All required dependencies are listed in requirements.txt. Install them using:

```bash
pip install -r requirements.txt
```

🔧 Dependencies Breakdown
Here’s a brief explanation of the key dependencies used:

## 🧠 Machine Learning & Deep Learning
tensorflow (2.18.0) – The core deep learning framework used to train and run the model.
keras (3.8.0) – High-level neural network API running on top of TensorFlow.
tensorflow-intel (2.18.0) – Optimized TensorFlow version for Intel-based systems.
## 🖼 Computer Vision
opencv-python (4.11.0.86) – Library for image processing and manipulation.
## 📊 Data Handling
numpy (2.0.2) – Supports matrix operations and numerical computations.
pandas (2.2.3) – Provides data structures for handling datasets efficiently.
ml-dtypes (0.4.1) – Numeric data types optimized for machine learning.
## 📉 Logging & Visualization
tensorboard (2.18.0) – Used for training visualization and performance tracking.
rich (13.9.4) – Provides beautifully formatted console output.
## 🌐 Networking & Requests
requests (2.32.3) – Simplifies making HTTP requests.
urllib3 (2.3.0) – Handles URL operations.
certifi (2025.1.31) – Provides SSL certificates for secure connections.
## 🔠 Others
h5py (3.12.1) – Used to store and retrieve trained models efficiently.
protobuf (5.29.3) – Handles structured data serialization.
werkzeug (3.1.3) – Essential for web-based applications (if extended for APIs).
## 🎯 Running the Model
Once dependencies are installed, run the script with:

```bash
python main.py
```
You need to have a webcam, but if you want to. You could use just an image
