import tkinter as tk
from tkinter import simpledialog, filedialog
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mplsoccer import Pitch
import matplotlib.animation as animation
import tkinter.simpledialog
import os

NUM_PLAYERS_PER_TEAM = 11
PITCH_LENGTH = 120
PITCH_WIDTH = 80
TOTAL_PLAYERS = NUM_PLAYERS_PER_TEAM * 2 + 1  # +1 for the ball


class FormationMovementEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Formation & Movement Editor")
        self.root.geometry("1690x1080")

        self.pitch = Pitch(pitch_type='statsbomb', pitch_color='grass', line_color='white')
        self.fig, self.ax = self.pitch.draw(figsize=(20, 9))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack()
        plt.close(self.fig)

        self.positions = np.zeros((TOTAL_PLAYERS, 2))
        self.roles = [""] * TOTAL_PLAYERS
        self.paths = {i: [] for i in range(TOTAL_PLAYERS)}
        self.selected_player = None
        self.mode = "formation"  # can be "formation" or "movement"

        self.dots = None
        self.cid_press = self.canvas.mpl_connect("button_press_event", self.on_press)
        self.cid_release = self.canvas.mpl_connect("button_release_event", self.on_release)
        self.cid_motion = self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.cid_click = self.canvas.mpl_connect("button_press_event", self.on_click)

        self.dragging_idx = None

        # GUI Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="Assign Roles", command=self.assign_roles).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Toggle Mode", command=self.toggle_mode).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Save to JSON", command=self.save_to_json).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Load Formation", command=self.load_formation).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Deselect Player", command=self.deselect_player).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Undo (Movement)", command=self.undo_last_point).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Play Animation", command=self.play_animation).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Reset", command=self.reset_positions).pack(side=tk.LEFT, padx=5)

        self.reset_positions()

        self.root.bind('<Return>', lambda event: self.deselect_player())

    def toggle_mode(self):
        self.selected_player = None
        self.mode = "movement" if self.mode == "formation" else "formation"
        msg = f"Switched to {self.mode.capitalize()} Mode"
        # Show temporary message
        if hasattr(self, 'mode_label') and self.mode_label:
            self.mode_label.destroy()
        self.mode_label = tk.Label(self.root, text=msg, font=("Arial", 16), bg="yellow")
        self.mode_label.place(relx=0.5, rely=0.02, anchor='n')
        self.root.after(1200, self.mode_label.destroy)
        print(msg)

    def reset_positions(self):
        # Line up blue team at x=20, red team at x=100, evenly spaced on y-axis
        y_positions = np.linspace(10, 70, NUM_PLAYERS_PER_TEAM)
        home_team = np.column_stack((np.full(NUM_PLAYERS_PER_TEAM, 20), y_positions))
        away_team = np.column_stack((np.full(NUM_PLAYERS_PER_TEAM, 100), y_positions))
        # Ball at fixed normalized position, denormalized to pitch coordinates
        ball_norm = [0.32214559386973174, 0.5037931034482759]
        ball = np.array([[ball_norm[0] * PITCH_LENGTH, ball_norm[1] * PITCH_WIDTH]])
        self.positions = np.vstack((home_team, away_team, ball))
        self.paths = {i: [] for i in range(TOTAL_PLAYERS)}  # Clear all lines
        self.roles = [""] * TOTAL_PLAYERS  # Clear all roles
        self.selected_player = None  # Deselect any player
        self.update_plot()

    def on_press(self, event):
        if self.mode != "formation" or event.inaxes != self.ax:
            return
        clicked_pos = np.array([event.xdata, event.ydata])
        dists = np.linalg.norm(self.positions - clicked_pos, axis=1)
        nearest_idx = np.argmin(dists)
        if dists[nearest_idx] < 1.5:
            self.dragging_idx = nearest_idx

    def on_motion(self, event):
        if self.mode != "formation" or self.dragging_idx is None or event.inaxes != self.ax:
            return
        self.positions[self.dragging_idx] = [event.xdata, event.ydata]
        self.update_plot()

    def on_release(self, event):
        self.dragging_idx = None

    def on_click(self, event):
        if self.mode != "movement" or event.inaxes != self.ax:
            return
        x, y = event.xdata, event.ydata
        clicked_pos = np.array([x, y])

        if self.selected_player is None:
            dists = np.linalg.norm(self.positions - clicked_pos, axis=1)
            self.selected_player = np.argmin(dists)
            print(f"Selected Player {self.selected_player + 1}")
        else:
            pid = self.selected_player
            self.paths[pid].append([x, y])
            if pid == TOTAL_PLAYERS - 1:
                color = 'black'
            else:
                color = 'blue' if pid < NUM_PLAYERS_PER_TEAM else 'red'
            if len(self.paths[pid]) > 1:
                self.ax.plot(*zip(*self.paths[pid]), color=color, linewidth=2)
                self.canvas.draw()

    def assign_roles(self):
        for i in range(TOTAL_PLAYERS):
            team = "Home (Blue)" if i < NUM_PLAYERS_PER_TEAM else "Away (Red)"
            number = (i % NUM_PLAYERS_PER_TEAM) + 1
            role = simpledialog.askstring(
                "Assign Role",
                f"Enter role for {team} Player {number} (e.g., LB, ST):",
                parent=self.root
            )
            if role:
                self.roles[i] = role
        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        self.pitch.draw(ax=self.ax)
        colors = ['blue' if i < NUM_PLAYERS_PER_TEAM else 'red' for i in range(TOTAL_PLAYERS)]
        self.dots = self.ax.scatter(self.positions[:, 0], self.positions[:, 1], s=300, color=colors,
                                    edgecolors='white', linewidth=2)
        for i, pos in enumerate(self.positions):
            if i == TOTAL_PLAYERS - 1:
                self.ax.scatter(pos[0], pos[1], s=200, color='black', edgecolors='white', linewidth=2, zorder=10)
                self.ax.text(pos[0], pos[1], "Ball", ha='center', va='center', fontsize=10, color='white', weight='bold')
            else:
                # Numbering: 1-11 for both teams
                number = (i % NUM_PLAYERS_PER_TEAM) + 1
                self.ax.text(pos[0], pos[1], str(number), ha='center', va='center', fontsize=12, color='white', weight='bold')
                # Draw role below dot (if assigned)
                if self.roles[i]:
                    self.ax.text(pos[0], pos[1] - 3, self.roles[i], ha='center', va='top', fontsize=10, color='yellow', weight='bold')
        # Re-draw existing paths
        for pid, path in self.paths.items():
            if len(path) > 1:
                if pid == TOTAL_PLAYERS - 1:
                    color = 'black'
                elif pid < NUM_PLAYERS_PER_TEAM:
                    color = 'blue'
                else:
                    color = 'red'
                self.ax.plot(*zip(*path), color=color, linewidth=2)
        self.canvas.draw()

    # In save_to_json, always use the fixed path for the ball
    def save_to_json(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".json", initialdir=".", filetypes=[("JSON files", "*.json")])
        if file_path:
            data = []
            for i in range(TOTAL_PLAYERS):
                if i == TOTAL_PLAYERS - 1:  # Ball
                    norm_path = [
                        [0.32214559386973174, 0.5037931034482759],
                        [0.9711877394636015, 0.5126436781609196]
                    ]
                else:
                    norm_path = [[float(pt[0]) / PITCH_LENGTH, float(pt[1]) / PITCH_WIDTH] for pt in self.paths[i]] if self.paths[i] else [[float(self.positions[i][0]) / PITCH_LENGTH, float(self.positions[i][1]) / PITCH_WIDTH]]
                player_data = {
                    "player_id": i + 1,
                    "role": self.roles[i],
                    "team": "Home" if i < NUM_PLAYERS_PER_TEAM else ("Away" if i < NUM_PLAYERS_PER_TEAM * 2 else "Ball"),
                    "is_ball": (i == TOTAL_PLAYERS - 1),
                    "path": norm_path
                }
                data.append(player_data)
            # Append metadata ONCE, after all players
            file_name = os.path.basename(file_path)  # Get the file name without the path")
            print(file_name)
            if "_" in file_name:
                formation = f'{"-".join(file_name.split("_")[0])}_{file_name.split(".json")[0].split("_")[1]}'
            else:
                formation = "-".join(file_name.split(".json")[0])
            # Ask user for style before saving
            def_style = tkinter.simpledialog.askstring("Defensive Style", "Enter defensive style (e.g., mid_press, low_block):", parent=self.root)
            if def_style is None:
                def_style = ""
            att_style = tkinter.simpledialog.askstring("Attacking Style", "Enter attacking style (e.g., wide_buildup, fast_counter):", parent=self.root)
            if att_style is None:
                att_style = ""
            width_style = tkinter.simpledialog.askstring("Width", "Enter width (e.g., narrow, wide):", parent=self.root)
            if width_style is None:
                width_style = ""

            style = {
                "defensive": def_style,
                "attacking": att_style,
                "width": width_style
            }

            metadata = {
                "metadata": {
                    "formation_home": formation,
                    "formation_away": formation,
                    "attacking_team": "Home",
                    "style": style
                }
            }
            data.append(metadata)
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Saved movement and roles to {file_path}")

    # In load_formation, always set the ball's path and position to the fixed path
    def load_formation(self):
        file_path = filedialog.askopenfilename(initialdir=".", filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, "r") as f:
                data = json.load(f)
            self.roles = [""] * TOTAL_PLAYERS
            self.positions = np.zeros((TOTAL_PLAYERS, 2))
            self.paths = {i: [] for i in range(TOTAL_PLAYERS)}
            for player in data:
                if isinstance(player, dict) and "metadata" in player:
                    continue
                idx = player["player_id"] - 1
                self.roles[idx] = player.get("role", "")
                if idx == TOTAL_PLAYERS - 1:  # Ball
                    fixed_path = [
                        [0.32214559386973174 * PITCH_LENGTH, 0.5037931034482759 * PITCH_WIDTH],
                        [0.9711877394636015 * PITCH_LENGTH, 0.5126436781609196 * PITCH_WIDTH]
                    ]
                    self.paths[idx] = fixed_path
                    self.positions[idx] = fixed_path[0]
                elif "path" in player and player["path"]:
                    denorm_path = [[pt[0] * PITCH_LENGTH, pt[1] * PITCH_WIDTH] for pt in player["path"]]
                    self.paths[idx] = denorm_path
                    self.positions[idx] = denorm_path[0]
            self.update_plot()

    def deselect_player(self):
        self.selected_player = None
        print("Deselected player.")

    def undo_last_point(self):
        if self.mode == "movement" and self.selected_player is not None:
            pid = self.selected_player
            if self.paths[pid]:
                self.paths[pid].pop()
                self.update_plot()
                print(f"Last point removed for Player {pid + 1}")
            else:
                print(f"No points to undo for Player {pid + 1}")

    def play_animation(self):
        SMOOTHNESS = 40
        PAUSE_FRAMES = 10

        # Separate blue and red team
        team_ids = [i for i in range(TOTAL_PLAYERS) if self.paths[i]]
        player_paths = {pid: np.array(self.paths[pid]) for pid in team_ids}
        player_roles = {pid: self.roles[pid] for pid in team_ids}
        player_numbers = {pid: str((pid % NUM_PLAYERS_PER_TEAM) + 1) for pid in team_ids}

        def interpolate_path(path, num_steps):
            if len(path) == 1:
                return np.repeat(path, num_steps, axis=0)
            interp_path = []
            for i in range(len(path) - 1):
                start = path[i]
                end = path[i + 1]
                steps = np.linspace(0, 1, num=int(num_steps / (len(path) - 1)), endpoint=False)
                for s in steps:
                    interp_point = start + (end - start) * s
                    interp_path.append(interp_point)
            interp_path.append(path[-1])
            return np.array(interp_path)

        # Interpolate all paths
        if not player_paths:
            print("No player paths to animate.")
            return
        max_steps = max(len(p) for p in player_paths.values())
        smooth_steps = max_steps * SMOOTHNESS
        interpolated_paths = {pid: interpolate_path(path, smooth_steps) for pid, path in player_paths.items()}

        # --- Create a new figure for animation ---
        pitch = Pitch(pitch_type='statsbomb', pitch_color='grass', line_color='white')
        fig, ax = pitch.draw(figsize=(12, 8))

        dots = []
        labels = []

        for pid in team_ids:
            initial = interpolated_paths[pid][0]
            color = 'blue' if pid < NUM_PLAYERS_PER_TEAM else 'red'
            number = (pid % NUM_PLAYERS_PER_TEAM) + 1
            role = player_roles[pid]
            # Dot
            dot = ax.scatter(initial[0], initial[1], s=300, color=color, edgecolors='white', linewidth=2)
            # Number inside dot
            num_label = ax.text(initial[0], initial[1], str(number), color='white', fontsize=12, ha='center', va='center', weight='bold')
            labels.append(num_label)
            # Role below dot (if assigned)
            if role:
                role_label = ax.text(initial[0], initial[1] - 3, role, color='yellow', fontsize=10, ha='center', va='top', weight='bold')
                labels.append(role_label)
            dots.append(dot)

        # Ball (23rd entity)
        if TOTAL_PLAYERS > NUM_PLAYERS_PER_TEAM * 2:
            ball_path = np.array(self.paths[TOTAL_PLAYERS - 1])
            if len(ball_path) > 1:
                ball_interp = interpolate_path(ball_path, smooth_steps)
            else:
                ball_interp = np.repeat(ball_path, smooth_steps, axis=0)
            ball_dot = ax.scatter(ball_interp[0][0], ball_interp[0][1], s=200, color='black', edgecolors='white', linewidth=2, zorder=10)
        else:
            ball_dot = None

        def animate(frame):
            label_idx = 0
            for i, pid in enumerate(team_ids):
                path = interpolated_paths[pid]
                pos = path[frame] if frame < len(path) else path[-1]
                dots[i].set_offsets(pos)
                # Move number label
                labels[label_idx].set_position(pos)
                label_idx += 1
                # Move role label if exists
                if player_roles[pid]:
                    labels[label_idx].set_position((pos[0], pos[1] - 3))
                    label_idx += 1
            # Animate ball
            if ball_dot is not None:
                pos = ball_interp[frame] if frame < len(ball_interp) else ball_interp[-1]
                ball_dot.set_offsets(pos)
                return dots + labels + [ball_dot]
            else:
                return dots + labels

        ani = animation.FuncAnimation(
            fig,
            animate,
            frames=smooth_steps + PAUSE_FRAMES,
            interval=300 // SMOOTHNESS,
            blit=True,
            repeat=True
        )

        plt.title("Animated Team Movement")
        plt.show()  # <-- This will pop out a new window


if __name__ == "__main__":
    root = tk.Tk()
    app = FormationMovementEditor(root)
    root.mainloop()

