# 🎭 Emotion Detection Web App

A Flask-based deep learning web application that detects human emotions from facial expressions using a pre-trained CNN model (fine-tuned on the FER2013 dataset).

---

## 🚀 Features

* Detects emotions such as **Happy, Sad, Angry, Surprise, Fear, Neutral, Disgust**
* Real-time detection from webcam or uploaded images
* Stores emotion analysis results in an SQLite database
* Lightweight and fast Flask web interface
* Supports retraining with custom datasets

---

## 🧠 Tech Stack

* **Frontend:** HTML5, CSS3, JavaScript
* **Backend:** Python (Flask)
* **Database:** SQLite
* **Machine Learning:** PyTorch (pre-trained model on FER2013)

---

## 📂 Project Structure

```
emotion_detection/
│
├── app.py              # Main Flask application
├── model.py            # CNN model structure
├── database.py         # SQLite database connection
├── index.html          # Frontend UI
├── requirements.txt    # Dependencies
├── emotionnet.pth      # Pre-trained model weights
├── static/             # Static assets
└── dataset/            # Optional custom training data
```

---

## ⚙️ Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/<your-username>/emotion-detection.git
cd emotion-detection
```

### 2️⃣ Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate     # On Windows
source venv/bin/activate  # On macOS/Linux
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the application

```bash
python app.py
```

Then open your browser and visit:

```
http://127.0.0.1:5000
```

---

## 🧩 Model Retraining (Optional)

To retrain the model with your dataset:

1. Place your images in the `dataset/images/` folder.
2. Update the `model.py` training function.
3. Save the new model as:

   ```
   emotion_retrain.pth
   ```
4. Replace the old model file in your project.

---

## 💾 Database

An SQLite database (`emotion_data.db`) stores:

* Detected emotions
* Timestamps
* Image file names (if saved)

---

## 📘 License

This project is open-source under the **MIT License**.

---

## 👤 Author

**Akoleaje Peter Taiwo Godsgift**
Built with 💙 using Flask & PyTorch
