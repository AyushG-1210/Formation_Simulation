import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mplsoccer import Pitch, FontManager

# Pitch size (StatsBomb default: 120x80, but you can use 105x68 if you prefer)
pitch_length = 105
pitch_width = 68

# Team A: 4-3-3
team_a_start = np.array([
    [0, 40], [18, 25], [18, 50], [18, 5], [18, 75], [32,12], [37, 40],
    [32,65], [50, 10], [50, 70], [52, 40]
])

# Team B: 4-4-2
team_b_start = np.array([
    [120, 40], [100, 50], [100, 25], [100, 6], [100, 75], [84, 50], [84, 25], [80, 6], 
    [80, 75], [65, 50], [65, 25]
])

# Passing sequence (indexes of players in team A)
pass_sequence = [1, 5, 6, 9, 10]  # GK → CM → RW → ST
ball_positions = team_a_start[pass_sequence]


# Target positions
team_a_target = team_a_start + np.array([[10, 0]] * len(team_a_start))
team_b_target = team_b_start - np.array([[10, 0]] * len(team_b_start))

all_outfield_start = np.vstack((team_a_start, team_b_start))
all_outfield_target = np.vstack((team_a_target, team_b_target))

# Set up mplsoccer pitch
pitch = Pitch(pitch_type='statsbomb', pitch_color='grass', line_color='white')
fig, ax = pitch.draw(figsize=(12, 8))
ax.set_title("Player Movement Simulation: 4-3-3 vs 4-4-2")

# Initial scatter for both teams
dots_a = pitch.scatter(team_a_start[:, 0], team_a_start[:, 1], ax=ax, s=300, color='blue', edgecolors='white', linewidth=2)
dots_b = pitch.scatter(team_b_start[:, 0], team_b_start[:, 1], ax=ax, s=300, color='red', edgecolors='white', linewidth=2)
# Use mplsoccer's built-in soccer ball marker
ball_dot = pitch.scatter(
    [ball_positions[0][0]], [ball_positions[0][1]],
    ax=ax, s=400, marker='o', color='white', edgecolors='black', linewidth=2, zorder=5
)
# Optionally, add a black pentagon to mimic a soccer ball pattern
# Draw pentagon AFTER the ball so it's on top (higher zorder)
ball_pentagon = pitch.scatter(
    [ball_positions[0][0]], [ball_positions[0][1]],
    ax=ax, s=80, marker='p', color='black', zorder=10  # zorder higher than ball_dot
)

def update_ball_pentagon(pos):
    ball_pentagon.set_offsets([pos])

# Modify animate to update the pentagon position
def animate(i):
    interp_pos = all_outfield_start + (all_outfield_target - all_outfield_start) * (i / 100)
    a_pos = interp_pos[:len(team_a_start)]
    b_pos = interp_pos[len(team_a_start):]
    dots_a.set_offsets(a_pos)
    dots_b.set_offsets(b_pos)
    
    # Ball movement logic
    segment_length = 100 // (len(pass_sequence) - 1)
    segment = min(i // segment_length, len(pass_sequence) - 2)
    seg_progress = (i % segment_length) / segment_length

    ball_start = ball_positions[segment]
    ball_end = ball_positions[segment + 1]
    ball_interp = ball_start + (ball_end - ball_start) * seg_progress
    ball_dot.set_offsets([ball_interp])
    update_ball_pentagon(ball_interp)

    return dots_a, dots_b, ball_dot, ball_pentagon

# Update init to clear pentagon
def init():
    dots_a.set_offsets(np.empty((0, 2)))
    dots_b.set_offsets(np.empty((0, 2)))
    ball_dot.set_offsets(np.empty((0, 2)))
    ball_pentagon.set_offsets(np.empty((0, 2)))
    return dots_a, dots_b, ball_dot, ball_pentagon

def animate(i):
    interp_pos = all_outfield_start + (all_outfield_target - all_outfield_start) * (i / 100)
    a_pos = interp_pos[:len(team_a_start)]
    b_pos = interp_pos[len(team_a_start):]
    dots_a.set_offsets(a_pos)
    dots_b.set_offsets(b_pos)
    
    # Ball movement logic
    segment_length = 100 // (len(pass_sequence) - 1)
    segment = min(i // segment_length, len(pass_sequence) - 2)
    seg_progress = (i % segment_length) / segment_length

    ball_start = ball_positions[segment]
    ball_end = ball_positions[segment + 1]
    ball_interp = ball_start + (ball_end - ball_start) * seg_progress
    ball_dot.set_offsets([ball_interp])

    return dots_a, dots_b, ball_dot


ani = animation.FuncAnimation(
    fig, animate, init_func=init, frames=101, interval=50, blit=True
)

plt.show()
