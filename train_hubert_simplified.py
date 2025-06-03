import os
import random
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from transformers import AutoModel
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

os.environ["TRANSFORMERS_OFFLINE"] = "1"

model_path = "/home/u213960/dades/hubert-ecg-small"
hubert_model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

print("Model loaded offline.")

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

# ------------------------------------
# Configuration and Constants
# ------------------------------------
# Update leads here to your chosen subset
LEADS = ['V1', 'V2', 'AVL']
LABEL_MAP = {"LCC": 0, "LVOTSUMMIT": 1}
BATCH_SIZE = 32
NUM_EPOCHS = 10
LR = 1e-4
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------------
# Load & Preprocess Data
# ------------------------------------
df = pd.read_pickle("ecg_dataset_preprocessed_fast.pkl")
df = df[df["Simplified"].isin(LABEL_MAP.keys())].copy()
df["label"] = df["Simplified"].map(LABEL_MAP)

# Patient-level split: 50% train, 15% val, 35% test
patients = df["PatientID"].unique()
train_p, test_p = train_test_split(patients, test_size=0.35, random_state=42)
train_p, val_p = train_test_split(train_p, test_size=0.23, random_state=42)  # 0.23 * 0.65 ≈ 0.15

train_df = df[df["PatientID"].isin(train_p)].reset_index(drop=True)
val_df = df[df["PatientID"].isin(val_p)].reset_index(drop=True)
test_df = df[df["PatientID"].isin(test_p)].reset_index(drop=True)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
print(f"Train Class Distribution Before SMOTE: {Counter(train_df['label'])}")

# ------------------------------------
# Extract Features & Apply SMOTE
# ------------------------------------
def prepare_features(df, leads):
    X = []
    y = []
    for _, row in df.iterrows():
        ecg = np.stack([row[lead][:500] for lead in leads])  # Shape: [C, T]
        X.append(ecg.flatten())
        y.append(row["label"])
    return np.stack(X), np.array(y)

X_train, y_train = prepare_features(train_df, LEADS)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Rebuild balanced train dataframe as tensors
train_ecgs = torch.tensor(X_resampled, dtype=torch.float32).reshape(-1, len(LEADS), 500)
train_labels = torch.tensor(y_resampled, dtype=torch.long)

# ------------------------------------
# Dataset Definition
# ------------------------------------
class ECGDataset(Dataset):
    def __init__(self, ecgs, labels):
        self.ecgs = ecgs
        self.labels = labels

    def __len__(self):
        return len(self.ecgs)

    def __getitem__(self, idx):
        return self.ecgs[idx], self.labels[idx]

def df_to_tensor_dataset(df, leads):
    ecgs, labels = [], []
    for _, row in df.iterrows():
        ecg = np.stack([row[lead][:500] for lead in leads])
        ecgs.append(torch.tensor(ecg, dtype=torch.float32))
        labels.append(torch.tensor(row["label"], dtype=torch.long))
    return ECGDataset(torch.stack(ecgs), torch.stack(labels))

train_set = ECGDataset(train_ecgs, train_labels)
val_set = df_to_tensor_dataset(val_df, LEADS)
test_set = df_to_tensor_dataset(test_df, LEADS)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

# ------------------------------------
# Load HuBERT-ECG Model
# ------------------------------------

# Freeze all but last 2 layers of encoder
for name, param in hubert_model.named_parameters():
    if "encoder.layers" in name:
        if any(layer_id in name for layer_id in ["10", "11"]):
            param.requires_grad = True
        else:
            param.requires_grad = False
    else:
        param.requires_grad = False

# ------------------------------------
# ECG Classifier Head
# ------------------------------------
class ECGClassifier(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(base_model.config.hidden_size, num_classes)

    def forward(self, x):  # x: [B, C, T]
        # Average across leads (C dimension)
        x = x.mean(dim=1)  # Shape needed for HuBERT input: [B, T]
        features = self.base_model(x).last_hidden_state  # [B, T, H]
        return self.classifier(features[:, 0, :])  # CLS token

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ECGClassifier(hubert_model, num_classes=2).to(device)

# ------------------------------------
# Loss, Optimizer
# ------------------------------------
class_weights = torch.tensor(
    [1.0 / np.sum(y_resampled == i) for i in range(2)],
    dtype=torch.float, device=device
)
criterion = CrossEntropyLoss(weight=class_weights)
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

# ------------------------------------
# Training Loop
# ------------------------------------
def evaluate(model, loader):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for ecg, label in loader:
            ecg, label = ecg.to(device), label.to(device)
            output = model(ecg)
            loss = criterion(output, label)
            total_loss += loss.item()
            all_preds.extend(output.argmax(dim=1).cpu().numpy())
            all_labels.extend(label.cpu().numpy())

    return total_loss / len(loader), all_preds, all_labels

train_losses, val_losses = [], []

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0

    for ecg, label in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        ecg, label = ecg.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(ecg)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    train_loss = running_loss / len(train_loader)
    val_loss, _, _ = evaluate(model, val_loader)

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")

# Save the model
torch.save(model.state_dict(), f"{RESULTS_DIR}/ecg_classifier_simplified.pth")
print(f"Model saved to {RESULTS_DIR}/ecg_classifier_simplified.pth")

# ------------------------------------
# Plot Loss Curve
# ------------------------------------
plt.figure()
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig(f"{RESULTS_DIR}/loss_curve_simplified.png")
plt.close()

# ------------------------------------
# Test Evaluation
# ------------------------------------
_, preds, labels = evaluate(model, test_loader)
acc = accuracy_score(labels, preds)
f1 = f1_score(labels, preds)
report = classification_report(labels, preds)
cm = confusion_matrix(labels, preds)

print(f"Test Accuracy: {acc:.4f}")
print(f"Test F1 Score: {f1:.4f}")
print("Classification Report:\n", report)
print("Confusion Matrix:\n", cm)

with open(f"{RESULTS_DIR}/test_report_simplified.txt", "w") as f:
    f.write(f"Test Accuracy: {acc:.4f}\n")
    f.write(f"Test F1 Score: {f1:.4f}\n\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    f.write(np.array2string(cm))

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABEL_MAP.keys(), yticklabels=LABEL_MAP.keys())
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/confusion_matrix_simplified.png")
plt.close()
