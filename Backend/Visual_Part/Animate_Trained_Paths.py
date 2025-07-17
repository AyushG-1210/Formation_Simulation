import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mplsoccer import Pitch
import matplotlib
matplotlib.use('TkAgg')

# === Load Data ===
X = np.load("C:\\Users\\anshu\\Desktop\\Simulation_Project\\Working_Data\\X_context.npy")           # (samples, 80, 4)
y = np.load("C:\\Users\\anshu\\Desktop\\Simulation_Project\\Working_Data\\y_context.npy")           # (samples, 20, 2)
model_preds = np.load("C:\\Users\\anshu\\Desktop\\Simulation_Project\\Working_Data\\model_predictions_with_ball.npy")  # (samples, 20, 2)

# === Denormalize ===
model_preds *= np.array([120, 80])
y *= np.array([120, 80])
X[:, :, :2] *= np.array([120, 80])  # only player positions

# === Settings ===
formation_id = 0  # 0–(num_formations - 1)
players_per_formation = 22
start_idx = formation_id * players_per_formation
end_idx = start_idx + players_per_formation

steps = 20
PAUSE_FRAMES = 10
total_frames = steps + PAUSE_FRAMES

# === Extract 22-player formation
player_inputs = X[start_idx:end_idx]       # (22, 80, 4)
true_paths = y[start_idx:end_idx]          # (22, 20, 2)
pred_paths = model_preds[start_idx:end_idx]  # (22, 20, 2)
starts = player_inputs[:, -1, :2]          # (22, 2)

# === Set up pitch
pitch = Pitch(pitch_type='statsbomb', pitch_color='grass', line_color='white')
fig, ax = pitch.draw(figsize=(12, 8))
ax.set_xlim(0, 120)
ax.set_ylim(0, 80)
ax.set_aspect('equal')

dots_pred = []
dots_true = []

for i in range(22):
    color = 'blue' if i < 11 else 'red'
    label_pred = "Predicted (Home)" if i == 0 else ("Predicted (Away)" if i == 11 else None)
    label_true = "Actual" if i == 0 else None

    dp, = ax.plot([], [], 'o', color=color, label=label_pred, markersize=7)
    da, = ax.plot([], [], 'x', color='lime', label=label_true, markersize=6)
    ax.text(starts[i][0], starts[i][1], str(i+1), color='white', fontsize=8, ha='center', va='center')

    dots_pred.append(dp)
    dots_true.append(da)

# === Animation Function
def update(frame):
    artists = []
    for i in range(22):
        if frame < PAUSE_FRAMES:
            dots_pred[i].set_data([starts[i][0]], [starts[i][1]])
            dots_true[i].set_data([starts[i][0]], [starts[i][1]])
        else:
            idx = frame - PAUSE_FRAMES
            if idx < steps:
                pred = pred_paths[i][idx]
                true = true_paths[i][idx]
                dots_pred[i].set_data([pred[0]], [pred[1]])
                dots_true[i].set_data([true[0]], [true[1]])
        artists.extend([dots_pred[i], dots_true[i]])
    return artists

# === Run Animation
ani = animation.FuncAnimation(
    fig,
    update,
    frames=total_frames,
    interval=300,
    blit=True,
    repeat=False
)

plt.title("Predicted vs Actual Movement (All 22 Players)")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
plt.tight_layout()
plt.show()
