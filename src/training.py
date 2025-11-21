import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


# Load CSV
df = pd.read_csv("/workspaces/HurricaneLiveDetection/Data/Processed/cleaned_martian_storms.csv")

# Separate features and target
target_col = "storm_occurrence"
features = [col for col in df.columns if col != target_col]
X = df[features].values
y = df[target_col].values

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Create sequences
sequence_length = 30
num_sequences = len(X_scaled) - sequence_length + 1
X_seq = np.zeros((num_sequences, sequence_length, X_scaled.shape[1]))
y_seq = np.zeros(num_sequences)

for i in range(num_sequences):
    X_seq[i] = X_scaled[i:i+sequence_length]
    y_seq[i] = y[i+sequence_length-1]

# Save as .npz
np.savez("/workspaces/HurricaneLiveDetection/Model/tcn_ready_dataset.npz", X=X_seq, y=y_seq)
print("TCN-ready dataset saved as tcn_ready_dataset.npz")
########################################################################################

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# -----------------------------
# 1. Load TCN-ready dataset
# -----------------------------
data = np.load("/workspaces/HurricaneLiveDetection/Model/tcn_ready_dataset.npz")
X, y = data["X"], data["y"]

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# -----------------------------
# 2. Train/Validation Split
# -----------------------------
train_size = int(0.8 * len(X_tensor))
val_size = len(X_tensor) - train_size
train_dataset, val_dataset = random_split(TensorDataset(X_tensor, y_tensor), [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

# -----------------------------
# 3. Define TCN Model
# -----------------------------
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(TCNBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=(kernel_size-1)*dilation, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out = self.conv(x)
        return self.dropout(self.relu(out))

class TCNModel(nn.Module):
    def __init__(self, num_features, num_classes=1):
        super(TCNModel, self).__init__()
        self.tcn1 = TCNBlock(num_features, 64, kernel_size=3, dilation=1)
        self.tcn2 = TCNBlock(64, 64, kernel_size=3, dilation=2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)
        x = self.tcn1(x)
        x = self.tcn2(x)
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)  # raw logits (no sigmoid)

# -----------------------------
# 4. Initialize Model, Loss, Optimizer
# -----------------------------
model = TCNModel(num_features=X.shape[2])

# Handle class imbalance
pos_weight = torch.tensor([len(y) / sum(y)])  # ratio of negatives to positives
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# 5. Training Loop with Validation & Metrics
# -----------------------------
epochs = 15
best_val_loss = float('inf')

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X).squeeze()
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    # Metrics
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    val_acc = (np.array(all_preds) == np.array(all_labels)).mean()

    print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss/len(train_loader):.4f} | "
          f"Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.4f}")
    print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1-score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "/workspaces/HurricaneLiveDetection/Model/best_tcn_model.pth")

print("Training complete. Best model saved as best_tcn_model.pth")