import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import datetime
import os
from torch import autocast  # Only import autocast

# === Parameters ===
MODEL_PATH = "team_model.pt"
BATCH_SIZE = 4
EPOCHS = 1200
PATIENCE = 200
LEARNING_RATE = 0.0005
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# === Load Data ===
X_players = np.load("X_players.npy")
X_ball    = np.load("X_ball.npy")
y_players = np.load("y_players.npy")

# === Train-test split ===
X_p_train, X_p_test, X_b_train, X_b_test, y_train, y_test = train_test_split(
    X_players, X_ball, y_players, test_size=0.2, random_state=SEED
)

# === Convert to Tensors ===
X_p_train = torch.tensor(X_p_train, dtype=torch.float32)
X_b_train = torch.tensor(X_b_train, dtype=torch.float32)
y_train   = torch.tensor(y_train, dtype=torch.float32)
X_p_test  = torch.tensor(X_p_test, dtype=torch.float32)
X_b_test  = torch.tensor(X_b_test, dtype=torch.float32)
y_test    = torch.tensor(y_test, dtype=torch.float32)

# === DataLoader ===
train_dataset = TensorDataset(X_p_train, X_b_train, y_train)
train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# === Model Definition ===
class TeamMovementModel(nn.Module):
    def __init__(self, player_dim=2, ball_dim=2, hidden_size=128):
        super().__init__()
        self.player_rnn = nn.LSTM(player_dim, hidden_size, batch_first=True)
        self.ball_rnn   = nn.LSTM(ball_dim, hidden_size, batch_first=True)
        self.fc         = nn.Linear(hidden_size * 2, 20 * 2)

    def forward(self, x_players, x_ball):
        B, P, T, _ = x_players.shape
        _, (ball_h, _) = self.ball_rnn(x_ball)
        ball_h = ball_h.squeeze(0)

        outputs = []
        for i in range(P):
            player_seq = x_players[:, i, :, :]
            _, (player_h, _) = self.player_rnn(player_seq)
            player_h = player_h.squeeze(0)

            combined = torch.cat([player_h, ball_h], dim=1)
            pred = self.fc(combined).view(-1, 20, 2)
            outputs.append(pred)

        return torch.stack(outputs, dim=1)  # (B, 22, 20, 2)

# === Training ===
model = TeamMovementModel()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_loss = float('inf')
no_improve = 0

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for xb_p, xb_b, yb in train_loader:
        optimizer.zero_grad()
        with autocast():  # Use autocast for mixed precision
            pred = model(xb_p, xb_b)
            loss = loss_fn(pred, yb)

        loss.backward()        # Standard backward
        optimizer.step()       # Standard optimizer step
        total_loss += loss.item() * xb_p.size(0)

    epoch_loss = total_loss / len(train_loader.dataset)

    # Test evaluation every 20 epochs
    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            test_pred = model(X_p_test, X_b_test)
            test_loss = loss_fn(test_pred, y_test).item()
        print(f"[{epoch}] Train Loss: {epoch_loss:.6f} | Test Loss: {test_loss:.6f}")
    elif epoch % 50 == 0:
        print(f"[{epoch}] Train Loss: {epoch_loss:.6f} | Time: {datetime.datetime.now().strftime('%H:%M:%S')}")


    # Early stopping
    if epoch_loss < best_loss - 1e-5:
        best_loss = epoch_loss
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

# === Save Model and Predictions ===
torch.save(model.state_dict(), os.path.join(MODEL_PATH))
print(f"Model saved to {MODEL_PATH}")

model.eval()
with torch.no_grad():
    final_pred = model(X_p_test, X_b_test)
    np.save(os.path.join("test_predictions.npy"), final_pred.numpy())
    print("Saved test predictions.")