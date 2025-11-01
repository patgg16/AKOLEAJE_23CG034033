import os
import io
import base64
from datetime import datetime
from PIL import Image
import numpy as np

from flask import Flask, render_template, request, jsonify, redirect, url_for
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker

import torch
import torch.nn as nn
from torchvision import transforms, models

# CONFIG
UPLOAD_FOLDER = os.path.join("static", "uploads")
MODEL_PATH = os.path.join("trained_models", "emotion_model.pth")
DB_PATH = "sqlite:///database.sqlite"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024  # 8MB

# Database setup (SQLAlchemy)
Base = declarative_base()
engine = create_engine(DB_PATH, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine)

class Usage(Base):
    __tablename__ = "usage"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, default="anonymous")
    filename = Column(String)
    result = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# Model loading
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path=MODEL_PATH, device=device):
    if not os.path.exists(model_path):
        print("Model file not found. Please train and place it at:", model_path)
        return None, None
    checkpoint = torch.load(model_path, map_location=device)
    # build same architecture as model.py used (resnet18)
    model = models.resnet18(pretrained=False)
    in_feat = model.fc.in_features
    # Try to infer num_classes from checkpoint (if available)
    if 'model_state_dict' in checkpoint:
        # assume final layer matches checkpoint
        # find length of saved fc or use placeholder
        model.fc = nn.Linear(in_feat, checkpoint['model_state_dict']['fc.weight'].shape[0]) \
            if 'fc.weight' in checkpoint['model_state_dict'] else nn.Linear(in_feat, 7)
        model.load_state_dict(checkpoint['model_state_dict'])
        class_to_idx = checkpoint.get('class_to_idx', None)
    else:
        # older save - attempt to load directly
        model.fc = nn.Linear(in_feat, 7)
        model.load_state_dict(checkpoint)
        class_to_idx = None
    model.eval()
    model.to(device)
    return model, class_to_idx

model, class_to_idx = load_model()

# Default inverse mapping if class_to_idx is not present
DEFAULT_CLASSES = ['angry','disgust','fear','happy','sad','surprise','neutral']

def preprocess_image_pil(pil_img):
    # convert to suitable tensor: grayscale->3 channel, resize to 48x48 or match training transforms
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((48,48)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    return transform(pil_img).unsqueeze(0)  # batch dim

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_image(pil_img):
    if model is None:
        return {"error":"Model not loaded"}
    input_tensor = preprocess_image_pil(pil_img).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        score, pred = torch.max(probs, 1)
        pred_idx = pred.item()
        # create inverse mapping
        if class_to_idx:
            idx_to_class = {v:k for k,v in class_to_idx.items()}
            label = idx_to_class.get(pred_idx, DEFAULT_CLASSES[pred_idx] if pred_idx < len(DEFAULT_CLASSES) else str(pred_idx))
        else:
            label = DEFAULT_CLASSES[pred_idx] if pred_idx < len(DEFAULT_CLASSES) else str(pred_idx)
        confidence = float(score.cpu().numpy())
    return {"label": label, "confidence": confidence}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    username = request.form.get("username", "anonymous")
    file = request.files.get("file")
    if not file or file.filename == "":
        return jsonify({"error":"no file provided"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error":"file type not allowed"}), 400
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    filename = f"{username}_{timestamp}_{file.filename}"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)
    # predict
    pil_img = Image.open(save_path).convert("RGB")
    result = predict_image(pil_img)
    # store in DB
    db = SessionLocal()
    usage = Usage(username=username, filename=filename, result=str(result))
    db.add(usage)
    db.commit()
    db.close()
    return jsonify({"filename": filename, "result": result})

@app.route("/capture", methods=["POST"])
def capture():
    """
    Expects JSON with 'image' field that contains base64 data URL from client (webcam)
    {'image': 'data:image/png;base64,...', 'username':'name'}
    """
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error":"no image provided"}), 400
    username = data.get("username", "anonymous")
    data_url = data['image']
    header, encoded = data_url.split(",", 1)
    imgdata = base64.b64decode(encoded)
    pil_img = Image.open(io.BytesIO(imgdata)).convert("RGB")
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    filename = f"{username}_{timestamp}.png"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    pil_img.save(save_path)
    result = predict_image(pil_img)
    # store in DB
    db = SessionLocal()
    usage = Usage(username=username, filename=filename, result=str(result))
    db.add(usage)
    db.commit()
    db.close()
    return jsonify({"filename": filename, "result": result})

@app.route("/history")
def history():
    db = SessionLocal()
    rows = db.query(Usage).order_by(Usage.created_at.desc()).limit(50).all()
    out = [{"id": r.id, "username": r.username, "filename": r.filename, "result": r.result, "created_at": r.created_at.isoformat()} for r in rows]
    db.close()
    return jsonify(out)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)