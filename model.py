import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader, random_split


DATA_DIR = "data"  
MODEL_DIR = "trained_models"
MODEL_FN = os.path.join(MODEL_DIR, "emotion_model.pth")
NUM_CLASSES = 7  
BATCH_SIZE = 64
NUM_EPOCHS = 10
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_transforms():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # convert to 3 channels if network expects 3
        transforms.Resize((48,48)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],  # ImageNet norms (works fine when fine-tuning)
                             [0.229, 0.224, 0.225])
    ])

def load_data():
    train_dir = os.path.join(DATA_DIR, "train")
    val_dir = os.path.join(DATA_DIR, "val")
    transform = get_transforms()
    train_ds = datasets.ImageFolder(train_dir, transform=transform)
    val_ds = datasets.ImageFolder(val_dir, transform=transform) if os.path.exists(val_dir) else None
    return train_ds, val_ds

def build_model(num_classes=NUM_CLASSES):
    # Use a pretrained model (resnet18) and fine-tune the last layer
    model = models.resnet18(pretrained=True)
    # adjust first conv if grayscale; we already convert to 3 channels so not required
    in_feat = model.fc.in_features
    model.fc = nn.Linear(in_feat, num_classes)
    return model

def train():
    os.makedirs(MODEL_DIR, exist_ok=True)
    train_ds, val_ds = load_data()
    if len(train_ds) == 0:
        raise ValueError("No training images found. Put images under data/train/<emotion>/")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4) if val_ds else None

    model = build_model().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_acc = 0.0
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data).item()

        epoch_loss = running_loss / len(train_ds)
        epoch_acc = running_corrects / len(train_ds)
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f}")

        # validate
        if val_loader:
            model.eval()
            val_corrects = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs = inputs.to(DEVICE)
                    labels = labels.to(DEVICE)
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    val_corrects += torch.sum(preds == labels.data).item()
            val_acc = val_corrects / len(val_ds)
            print(f"Validation acc: {val_acc:.4f}")
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'class_to_idx': train_ds.class_to_idx
                }, MODEL_FN)
                print(f"Saved best model to {MODEL_FN}")

    # if no val set, save final model
    if not val_loader:
        torch.save({
            'model_state_dict': model.state_dict(),
            'class_to_idx': train_ds.class_to_idx
        }, MODEL_FN)
        print(f"Saved model to {MODEL_FN}")

if __name__ == "__main__":
    train()