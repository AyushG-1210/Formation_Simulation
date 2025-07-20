import os
import json
import numpy as np
from scipy.interpolate import interp1d

BASE_DIR = "C:\\Users\\anshu\\Desktop\\Simulation_Project\\Backend"
data_dir = os.path.join(BASE_DIR, "Formations")
save_dir = os.path.join(BASE_DIR, "Working_Data")

# Define your roles (customize as needed)
ROLE_LIST = ['GK', 'CB', 'LB', 'RB', 'CM', 'CDM', 'CAM', 'LM', 'RM', 'LW', 'RW', 'ST','LWB','RWB']
role_to_idx = {role: i for i, role in enumerate(ROLE_LIST)}

X = []
y = []

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

    # Get ball path for this sample (if you want to use a fixed one, keep as is)
    ball_in = ball_path[0]  # Use the starting ball position

    players = [p for p in data if isinstance(p, dict) and p.get("is_ball") is not True and "player_id" in p and "role" in p]

    for player in players:
        path = np.array(player["path"])
        if path.shape[0] < 2 or path.shape[1] != 2:
            print(f"Skipping malformed player in {filename} (shape: {path.shape})")
            continue

        # Interpolate to 100 points
        old_steps = np.linspace(0, 1, path.shape[0])
        new_steps = np.linspace(0, 1, 100)
        interp_func = interp1d(old_steps, path, axis=0)
        interp_path = interp_func(new_steps)  # shape (100, 2)

        # Jitter
        jitter = np.random.uniform(-0.02, 0.02, size=2)
        start_pos = interp_path[0] + jitter
        start_pos = np.clip(start_pos, 0, 1)
        end_pos = interp_path[-1]
        role = player.get("role", "UNK")
        if role not in role_to_idx:
            print(f"Unknown role '{role}' in {filename}, skipping...")
            continue
        role_idx = role_to_idx.get(role, 0)

        # Input: [start_x, start_y, ball_x, ball_y, role_idx]
        X.append(np.concatenate([start_pos, ball_in, [role_idx]]))
        y.append(end_pos)

        # --- Mirrored sample ---
        mirrored_start = np.array([1 - start_pos[0], start_pos[1]])
        mirrored_end = np.array([1 - end_pos[0], end_pos[1]])

        # Optionally, swap left/right roles here if desired
        mirror_map = {'LB': 'RB', 'RB': 'LB', 'LWB': 'RWB', 'RWB': 'LWB', 'LM': 'RM', 'RM': 'LM', 'LW': 'RW', 'RW': 'LW'}
        mirrored_role = mirror_map.get(role, role)
        mirrored_role_idx = role_to_idx.get(mirrored_role, 0)
        X.append(np.concatenate([mirrored_start, ball_in, [mirrored_role_idx]]))
        y.append(mirrored_end)

X = np.array(X, dtype=np.float32)  # shape (N, 5)
y = np.array(y, dtype=np.float32)  # shape (N, 2)

print(f"Final shapes: X {X.shape}, y {y.shape}")

# Save datasets
np.save(os.path.join(save_dir, "X.npy"), X)
np.save(os.path.join(save_dir, "y.npy"), y)
