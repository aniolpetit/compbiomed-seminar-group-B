import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from transformers import AutoModel
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss

# Create results directory
os.makedirs("results", exist_ok=True)

# --------------------------
# 0. Load and Subsample Data
# --------------------------
df = pd.read_pickle('ecg_dataset_preprocessed_fast.pkl')

label_map = {'Left': 0, 'Right': 1}
df["label"] = df["Side"].map(label_map)

# Patient-level split
unique_patients = df["PatientID"].unique()
train_p, val_p = train_test_split(unique_patients, test_size=0.3, random_state=42)
val_p, test_p = train_test_split(val_p, test_size=0.5, random_state=42)

train_df = df[df["PatientID"].isin(train_p)].reset_index(drop=True)
val_df = df[df["PatientID"].isin(val_p)].reset_index(drop=True)
test_df = df[df["PatientID"].isin(test_p)].reset_index(drop=True)

# Soft downsample + reduce size
minority_class = train_df['label'].value_counts().idxmin()
majority_class = train_df['label'].value_counts().idxmax()

minority_df = train_df[train_df['label'] == minority_class]
majority_df = train_df[train_df['label'] == majority_class]

target_majority_size = int(len(minority_df) * 1.1)
majority_df_downsampled = majority_df.sample(n=target_majority_size, random_state=42)

# Combine and shuffle
train_df = pd.concat([minority_df, majority_df_downsampled]).sample(frac=1.0, random_state=42).reset_index(drop=True)

# Reduce both classes by half (equal ratio)
final_df = []
for label in [0, 1]:
    class_df = train_df[train_df["label"] == label]
    reduced = class_df.sample(frac=0.3, random_state=42)
    final_df.append(reduced)

train_df = pd.concat(final_df).sample(frac=1.0, random_state=42).reset_index(drop=True)

print(f"Train: {len(train_df)} | Class Distribution: {Counter(train_df['label'])}")
print(f"Val: {len(val_df)}, Test: {len(test_df)}")

# --------------------------
# 1. Define Dataset & Preprocessing
# --------------------------
class ECGDataset(Dataset):
    def __init__(self, dataframe, lead_names):
        self.dataframe = dataframe
        self.lead_names = lead_names

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        ecg_np = np.stack([row[lead][:500] for lead in self.lead_names], axis=0)
        ecg_tensor = torch.tensor(ecg_np, dtype=torch.float32)
        label = torch.tensor(row["label"], dtype=torch.long)
        return ecg_tensor, label

lead_names = ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

train_dataset = ECGDataset(train_df, lead_names)
val_dataset = ECGDataset(val_df, lead_names)
test_dataset = ECGDataset(test_df, lead_names)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

# --------------------------
# 2. Load Pretrained HuBERT-ECG
# --------------------------
hubert_model = AutoModel.from_pretrained("Edoardo-BS/hubert-ecg-small", trust_remote_code=True)

# Freeze all except classifier + last 2 encoder layers
for name, param in hubert_model.named_parameters():
    if "encoder.layers" in name:
        if any(layer_id in name for layer_id in ["10", "11"]):
            param.requires_grad = True
        else:
            param.requires_grad = False
    else:
        param.requires_grad = False

# --------------------------
# 3. Classification Head
# --------------------------
class ECGClassifier(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Linear(base_model.config.hidden_size, num_classes)

    def forward(self, x):  # x: [B, 12, 500]
        selected_leads = [0, 6, 7, 8]  # I, V1, V2, V3
        x = x[:, selected_leads, :].mean(dim=1)
        outputs = self.base_model(x).last_hidden_state
        cls_token = outputs[:, 0, :]
        return self.classifier(cls_token)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ECGClassifier(hubert_model, num_classes=2).to(device)

# --------------------------
# 4. Loss, Optimizer
# --------------------------
counts = Counter(train_df["label"])
class_weights = [1.0 / counts[0], 1.0 / counts[1]]
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

criterion = CrossEntropyLoss(weight=class_weights)
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# --------------------------
# 5. Training Loop
# --------------------------
train_losses, val_losses = [], []

for epoch in range(10):
    model.train()
    epoch_train_loss = 0.0
    for ecg, label in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        ecg, label = ecg.to(device), label.to(device)
        optimizer.zero_grad()
        output = model(ecg)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()

    avg_train_loss = epoch_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    epoch_val_loss = 0.0
    with torch.no_grad():
        for ecg, label in val_loader:
            ecg, label = ecg.to(device), label.to(device)
            output = model(ecg)
            loss = criterion(output, label)
            epoch_val_loss += loss.item()

    avg_val_loss = epoch_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# Save model
torch.save(model.state_dict(), "results/ecg_classifier_side.pth")
print("Model saved to results/ecg_classifier_side.pth")

# --------------------------
# 6. Save Loss Curve
# --------------------------
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("results/loss_curve_side_half.png")
plt.close()

# --------------------------
# 7. Test Evaluation
# --------------------------
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for ecg, label in test_loader:
        ecg, label = ecg.to(device), label.to(device)
        output = model(ecg)
        preds = torch.argmax(output, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(label.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
report = classification_report(all_labels, all_preds, target_names=["Left", "Right"])
cm = confusion_matrix(all_labels, all_preds)

print("Test Accuracy:", acc)
print("Test F1 Score:", f1)
print("\nClassification Report:\n", report)

# Save confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Left", "Right"], yticklabels=["Left", "Right"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("results/confusion_matrix_side_half.png")
plt.close()

# Save classification report
with open("results/classification_report_side_half.txt", "w") as f:
    f.write(f"Test Accuracy: {acc:.4f}\n")
    f.write(f"Test F1 Score: {f1:.4f}\n\n")
    f.write(report)
