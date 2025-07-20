import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

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

# === Load model ===
model = TeamMovementModel()
model.load_state_dict(torch.load("team_model.pt", map_location="cpu"))
model.eval()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if len(data) != 22:
            return jsonify({"error": "Expected 22 players"}), 400

        paths = [p["path"] for p in data]  # (22, T, 2) but T might be 1
        paths = np.array(paths).astype(np.float32)  # (22, T, 2)

        # --- NEW LOGIC: use only the first position, repeat for 20 timesteps ---
        start_positions = torch.tensor(paths[:, 0:1, :], dtype=torch.float32)  # (22, 1, 2)
        repeated_input = start_positions.expand(-1, 20, -1)  # (22, 20, 2)
        repeated_input = repeated_input.unsqueeze(0)  # (1, 22, 20, 2)
        dummy_ball = torch.zeros((1, 20, 2))  # (1, 20, 2)

        with torch.no_grad():
            preds = model(repeated_input, dummy_ball)  # (1, 22, 20, 2)

        preds = preds.squeeze(0).numpy()  # (22, 20, 2)

        # Prepare response: [ [start, end], ... ]
        result = []
        for i in range(22):
            start = paths[i, 0].tolist()
            end = preds[i, -1].tolist()  # last predicted point
            result.append([start, end])
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
