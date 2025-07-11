import os
import json
import numpy as np
from scipy.interpolate import interp1d

data_dir = "C:\\Users\\anshu\\Desktop\\Simulation_Project\\Formations"  # Your folder with JSON files
save_dir = "C:\\Users\\anshu\\Desktop\\Simulation_Project\\Working_Data"  # Save location

X_players = []
X_ball = []
y_players = []

# Define the fixed ball path (interpolated to 100 steps)
ball_start = np.array([0.32214559386973174, 0.5037931034482759])
ball_end = np.array([0.9711877394636015, 0.5126436781609196])
ball_path = np.linspace(ball_start, ball_end, 100)  # (100, 2)

# Loop through all formation JSONs
for filename in os.listdir(data_dir):
    if not filename.endswith(".json"):
        continue

    with open(os.path.join(data_dir, filename), "r") as f:
        data = json.load(f)

    players = [p for p in data if isinstance(p, dict) and p.get("is_ball") is not True and "player_id" in p]

    player_paths = []
    player_targets = []
    
    for player in players:
        path = np.array(player["path"])  # shape (N, 2)
        if path.shape[0] < 2 or path.shape[1] != 2:
            print(f"Skipping malformed player in {filename} (shape: {path.shape})")
            continue
        
        # Interpolate to 100 points
        old_steps = np.linspace(0, 1, path.shape[0])
        new_steps = np.linspace(0, 1, 100)
        interp_func = interp1d(old_steps, path, axis=0)
        interp_path = interp_func(new_steps)  # shape (100, 2)
        
        player_in = interp_path[:80]   # (80, 2)
        player_out = interp_path[80:]  # (20, 2)
        player_paths.append(player_in)
        player_targets.append(player_out)

    # Stack all player paths (22 players, each with 80 time steps, 2 dimensions)
    player_paths = np.array(player_paths)  # (22, 80, 2)
    player_targets = np.array(player_targets)  # (22, 20, 2)

    # Ball path (fixed, same for all players)
    ball_in = ball_path[:80]  # (80, 2)

    # Append to main datasets
    X_players.append(player_paths)
    X_ball.append(ball_in)
    y_players.append(player_targets)

# Convert lists to arrays
X_players = np.array(X_players)  # shape (samples, 22, 80, 2)
X_ball = np.array(X_ball)        # shape (samples, 80, 2)
y_players = np.array(y_players)  # shape (samples, 22, 20, 2)

print(f"Final shapes: X_players {X_players.shape}, X_ball {X_ball.shape}, y_players {y_players.shape}")

# Save datasets
np.save(os.path.join(save_dir, "X_players.npy"), X_players)
np.save(os.path.join(save_dir, "X_ball.npy"), X_ball)
np.save(os.path.join(save_dir, "y_players.npy"), y_players)
