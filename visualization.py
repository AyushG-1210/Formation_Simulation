import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import requests
from mplsoccer import Pitch

# --- 1. Pick a random formation file and load player paths ---
formations_dir = os.path.join("Backend", "Formations")
files = [f for f in os.listdir(formations_dir) if f.endswith(".json")]
chosen_file = random.choice(files)
with open(os.path.join(formations_dir, chosen_file), "r") as f:
    data = json.load(f)

# Extract player entries for player_id 1-22 with a non-empty path
player_entries = [
    p for p in data
    if isinstance(p, dict)
    and "player_id" in p
    and 1 <= p["player_id"] <= 22
    and "path" in p
    and isinstance(p["path"], list)
    and len(p["path"]) > 0
]

# Prepare input: only the first coordinate from each player's path
input_points = [p["path"][0] for p in sorted(player_entries, key=lambda x: x["player_id"])]

T = 20  # Set to the number of timesteps your model expects
payload = [{"path": [pt]*T} for pt in input_points]

# --- 2. Send to /predict endpoint ---
response = requests.post(
    "http://127.0.0.1:5000/predict",
    json=payload
)

print("Status code:", response.status_code)
if response.status_code != 200:
    print("Error:", response.text)
    exit()

predicted = np.array(response.json())  # shape: (22, T, 2)

# --- 3. Visualize: football pitch style like Interface+Animation.py ---

NUM_PLAYERS_PER_TEAM = 11
PITCH_LENGTH = 120
PITCH_WIDTH = 80

pitch = Pitch(pitch_type='statsbomb', pitch_color='grass', line_color='white')
fig, ax = pitch.draw(figsize=(12, 8))

for i, player_traj in enumerate(predicted):
    x = np.array(player_traj)[[0, 1], 0] * PITCH_LENGTH
    y = np.array(player_traj)[[0, 1], 1] * PITCH_WIDTH
    color = 'blue' if i < NUM_PLAYERS_PER_TEAM else 'red'
    # Draw trajectory
    ax.plot(x, y, color=color, linewidth=2, zorder=2)
    # Draw starting dot with white outline
    ax.scatter(x[0], y[0], s=300, color=color, edgecolors='white', linewidth=2, zorder=3)
    # Draw number in white
    number = (i % NUM_PLAYERS_PER_TEAM) + 1
    ax.text(x[0], y[0], str(number), color='white', fontsize=12, ha='center', va='center', weight='bold', zorder=4)

# Optionally, plot the input points as large dots (to show where the input was)
input_x = [pt[0] for pt in input_points]
input_y = [pt[1] for pt in input_points]
ax.scatter(input_x[:NUM_PLAYERS_PER_TEAM], input_y[:NUM_PLAYERS_PER_TEAM], color='blue', s=400, edgecolors='white', linewidth=2, alpha=0.3, zorder=1)
ax.scatter(input_x[NUM_PLAYERS_PER_TEAM:], input_y[NUM_PLAYERS_PER_TEAM:], color='red', s=400, edgecolors='white', linewidth=2, alpha=0.3, zorder=1)

plt.title(f'Predicted Player Trajectories from {chosen_file}')
plt.tight_layout()
plt.show()