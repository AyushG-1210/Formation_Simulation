from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import interp1d
import os
import json

app = FastAPI()

# CORS (adjust allowed origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path setup
FORMATIONS_DIR = "Backend/Formations"
MODEL_PATH = "Backend/ML_Part/team_model.pt"

# ----------- Model Definition -----------
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


def load_model():
    model = TeamMovementModel()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    return model


# ----------- Interpolation Logic -----------
def preprocess_formation(json_data):
    players = [p for p in json_data if isinstance(p, dict) and not p.get("is_ball") and "player_id" in p]

    ball_start = np.array([0.32214559386973174, 0.5037931034482759])
    ball_end = np.array([0.9711877394636015, 0.5126436781609196])
    ball_path = np.linspace(ball_start, ball_end, 100)[:80]  # (80, 2)

    player_inputs = []
    for player in players:
        path = np.array(player["path"])
        if path.shape[0] < 2 or path.shape[1] != 2:
            raise ValueError("Malformed path for player")

        old_steps = np.linspace(0, 1, path.shape[0])
        new_steps = np.linspace(0, 1, 100)
        interp_func = interp1d(old_steps, path, axis=0)
        interp_path = interp_func(new_steps)

        player_inputs.append(interp_path[:80])  # (80, 2)

    x_players = np.array(player_inputs)  # (22, 80, 2)
    x_ball = ball_path  # (80, 2)
    return x_players, x_ball


# ----------- API Endpoints -----------

@app.post("/save-formation")
async def save_formation(request: Request):
    try:
        data = await request.json()
        filename = f"formation_{len(os.listdir(FORMATIONS_DIR)) + 1}.json"
        filepath = os.path.join(FORMATIONS_DIR, filename)
        with open(filepath, "w") as f:
            json.dump(data, f)
        return {"message": "Formation saved", "file": filename}
    except Exception as e:
        return {"error": str(e)}


@app.post("/predict")
async def predict_formation(request: Request):
    try:
        json_data = await request.json()
        model = load_model()

        x_players, x_ball = preprocess_formation(json_data)
        x_players_tensor = torch.tensor(x_players, dtype=torch.float32).unsqueeze(0)  # (1, 22, 80, 2)
        x_ball_tensor = torch.tensor(x_ball, dtype=torch.float32).unsqueeze(0)        # (1, 80, 2)

        with torch.no_grad():
            pred = model(x_players_tensor, x_ball_tensor)  # (1, 22, 20, 2)
            pred_np = pred.squeeze(0).numpy().tolist()

        return {"predicted_paths": pred_np}

    except Exception as e:
        return {"error": str(e)}
