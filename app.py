import gradio as gr
import torch
import numpy as np

# -----------------------------
# Load trained TCN model
# -----------------------------
class TCNBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size,
                                    padding=(kernel_size-1)*dilation, dilation=dilation)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.2)

    def forward(self, x):
        out = self.conv(x)
        return self.dropout(self.relu(out))

class TCNModel(torch.nn.Module):
    def __init__(self, num_features, num_classes=1):
        super().__init__()
        self.tcn1 = TCNBlock(num_features, 64, kernel_size=3, dilation=1)
        self.tcn2 = TCNBlock(64, 64, kernel_size=3, dilation=2)
        self.global_pool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc = torch.nn.Linear(64, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (batch, features, seq_len)
        x = self.tcn1(x)
        x = self.tcn2(x)
        x = self.global_pool(x).squeeze(-1)
        return self.fc(x)

# Load model weights
model = TCNModel(num_features=11)
model.load_state_dict(torch.load("/workspaces/HurricaneLiveDetection/Model/best_tcn_model.pth", map_location=torch.device('cpu')))
model.eval()

# -----------------------------
# Prediction function
# -----------------------------
def predict(Ls, month, martian_year, latitude, longitude, min_temp, max_temp, pressure, confidence, missing_data, mission_phase):
    # Create feature array
    features = np.array([Ls, month, martian_year, latitude, longitude, min_temp, max_temp, pressure, confidence, missing_data, mission_phase], dtype=np.float32)
    
    # Repeat across sequence length for TCN
    seq_len = 30
    X = np.tile(features, (seq_len, 1))  # shape (seq_len, features)
    X = torch.tensor(X).unsqueeze(0)     # shape (1, seq_len, features)

    with torch.no_grad():
        prob = torch.sigmoid(model(X)).item()
    return f"Prediction: {'Storm' if prob >= 0.5 else 'No Storm'} (Confidence: {prob:.2f})"

# -----------------------------
# Gradio Interface
# -----------------------------
interface = gr.Interface(
    fn=predict,
    inputs=[
        gr.Slider(0, 360, step=1, label="Ls"),
        gr.Slider(1, 12, step=1, label="Month"),
        gr.Slider(30, 35, step=1, label="Martian Year"),
        gr.Slider(-90, 90, step=0.1, label="Centroid Latitude"),
        gr.Slider(-180, 180, step=0.1, label="Centroid Longitude"),
        gr.Slider(-90, -60, step=0.5, label="Min Temp (°C)"),
        gr.Slider(-25, 0, step=0.5, label="Max Temp (°C)"),
        gr.Slider(730, 920, step=1, label="Pressure (Pa)"),
        gr.Slider(50, 100, step=1, label="Confidence Interval"),
        gr.Dropdown(choices=[0, 1], label="Missing Data"),
        gr.Slider(0, 15, step=1, label="Mission Subphase")
    ],
    outputs="text",
    title="Martian Dust Storm Predictor (Interactive Demo)",
    description="Adjust the sliders and dropdowns to simulate conditions and predict storm occurrence."
)

interface.launch()