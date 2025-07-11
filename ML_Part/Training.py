import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import datetime

# === Load Data ===
X_players = np.load("Working_Data/X_players.npy")   # (samples, 22, 80, 2)
X_ball = np.load("Working_Data/X_ball.npy")         # (samples, 80, 2)
y_players = np.load("Working_Data/y_players.npy")   # (samples, 22, 20, 2)

# Train-test split
X_p_train, X_p_test, X_b_train, X_b_test, y_train, y_test = train_test_split(
    X_players, X_ball, y_players, test_size=0.2, random_state=42
)

# Convert to PyTorch tensors
X_p_train = torch.tensor(X_p_train, dtype=torch.float32)
X_b_train = torch.tensor(X_b_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

X_p_test = torch.tensor(X_p_test, dtype=torch.float32)
X_b_test = torch.tensor(X_b_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# === DataLoader ===
batch_size = 4

train_dataset = TensorDataset(X_p_train, X_b_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# === Model Definition ===
class TeamMovementModel(nn.Module):
    def __init__(self, player_dim=2, ball_dim=2, hidden_size=128):
        super().__init__()
        self.player_rnn = nn.LSTM(input_size=player_dim, hidden_size=hidden_size, batch_first=True)
        self.ball_rnn = nn.LSTM(input_size=ball_dim, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, 20 * 2)  # Predict 20 steps (x, y) for each player

    def forward(self, x_players, x_ball):
        batch_size, num_players, seq_len, _ = x_players.shape
        outputs = []

        _, (ball_h, _) = self.ball_rnn(x_ball)  # (1, B, H)
        ball_h = ball_h.squeeze(0)  # (B, H)

        for i in range(num_players):
            player_input = x_players[:, i, :, :]  # (B, 80, 2)
            _, (player_h, _) = self.player_rnn(player_input)  # (1, B, H)
            player_h = player_h.squeeze(0)  # (B, H)

            combined = torch.cat([player_h, ball_h], dim=1)  # (B, 2H)
            out = self.fc(combined).view(-1, 20, 2)  # (B, 20, 2)
            outputs.append(out)

        return torch.stack(outputs, dim=1)  # (B, 22, 20, 2)

# === Initialize ===
model = TeamMovementModel()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# === Training Loop ===
epochs = 1200
patience = 200
best_loss = float('inf')
epochs_no_improve = 0

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for batch_x_p, batch_x_b, batch_y in train_loader:
        pred = model(batch_x_p, batch_x_b)
        loss = loss_fn(pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_x_p.size(0)

    epoch_loss = running_loss / len(train_loader.dataset)

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {epoch_loss:.5f}, Time: {datetime.datetime.now().strftime(' %H:%M:%S')}")

    if epoch_loss < best_loss - 1e-5:
        best_loss = epoch_loss
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(f"Early stopping at epoch {epoch}, best loss: {best_loss:.6f}")
        break

# === Save Model ===
torch.save(model.state_dict(), "ML_Part/team_model.pt")
print("Model saved to team_model.pt")

# === Evaluate on Test Sample ===
model.eval()
with torch.no_grad():
    test_pred = model(X_p_test, X_b_test)  # (N, 22, 20, 2)
    np.save("Working_Data/test_predictions.npy", test_pred.numpy())
    print("Saved predictions to test_predictions.npy")

# need to add test_loss check fro every 20 epochs and move the model to google colab for better training