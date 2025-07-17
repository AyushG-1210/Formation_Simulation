import tkinter as tk
from tkinter import simpledialog, filedialog
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mplsoccer import Pitch

# Constants
NUM_PLAYERS = 11
PITCH_LENGTH = 120
PITCH_WIDTH = 80

class FormationEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Football Formation Editor")
        self.root.geometry("1690x1080")  # Set the window size (width x height)

        # Setup matplotlib pitch
        self.pitch = Pitch(pitch_type='statsbomb', pitch_color='grass', line_color='white')
        self.fig, self.ax = self.pitch.draw(figsize=(20, 9))  # Increase figure size
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack()

        # Player data
        self.positions = np.zeros((NUM_PLAYERS, 2))
        self.roles = [""] * NUM_PLAYERS
        self.selected = None
        self.dragging_idx = None  # Track which dot is being dragged
        self.dots = self.ax.scatter([], [], s=300, color='blue', edgecolors='white', linewidth=2)

        # Bind events for dragging
        self.cid_press = self.canvas.mpl_connect("button_press_event", self.on_press)
        self.cid_release = self.canvas.mpl_connect("button_release_event", self.on_release)
        self.cid_motion = self.canvas.mpl_connect("motion_notify_event", self.on_motion)

        # Buttons
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="Assign Roles", command=self.assign_roles).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Save Formation", command=self.save_formation).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Load Formation", command=self.load_formation).pack(side=tk.LEFT, padx=5)

        # Initialize default positions
        self.reset_positions()

    def reset_positions(self):
        x_coords = np.linspace(20, 100, NUM_PLAYERS)
        y_coords = np.linspace(10, 70, NUM_PLAYERS)
        self.positions = np.column_stack((x_coords, y_coords))
        self.update_plot()

    def on_press(self, event):
        if event.inaxes != self.ax:
            return
        clicked_pos = np.array([event.xdata, event.ydata])
        dists = np.linalg.norm(self.positions - clicked_pos, axis=1)
        nearest_idx = np.argmin(dists)
        # Only start dragging if click is close to a dot (within 1.5 units)
        if dists[nearest_idx] < 1.5:
            self.dragging_idx = nearest_idx

    def on_motion(self, event):
        if self.dragging_idx is not None and event.inaxes == self.ax:
            self.positions[self.dragging_idx] = [event.xdata, event.ydata]
            self.update_plot()

    def on_release(self, event):
        self.dragging_idx = None

    def assign_roles(self):
        for i in range(NUM_PLAYERS):
            role = simpledialog.askstring("Assign Role", f"Enter role for Player {i + 1} (e.g., LB, ST):", parent=self.root)
            if role:
                self.roles[i] = role
        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        self.pitch.draw(ax=self.ax)
        self.dots = self.ax.scatter(self.positions[:, 0], self.positions[:, 1], s=300, color='blue', edgecolors='white', linewidth=2)
        for i, pos in enumerate(self.positions):
            label = self.roles[i] if self.roles[i] else str(i + 1)
            self.ax.text(pos[0], pos[1], label, ha='center', va='center', fontsize=10, color='white')
        self.canvas.draw()

    def save_formation(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".json", initialdir="saved_formations", filetypes=[("JSON files", "*.json")])
        if file_path:
            data = []
            for i in range(NUM_PLAYERS):
                player_data = {
                    "player_id": i + 1,
                    "role": self.roles[i],
                    "x": float(self.positions[i, 0]),
                    "y": float(self.positions[i, 1])
                }
                data.append(player_data)
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Formation saved to {file_path}")

    def load_formation(self):
        file_path = filedialog.askopenfilename(initialdir="saved_formations", filetypes=[("JSON files", "*.json")])
        if file_path:
            with open(file_path, "r") as f:
                data = json.load(f)
            # data is a list of dicts: {"player_id", "role", "x", "y"}
            self.positions = np.zeros((NUM_PLAYERS, 2))
            self.roles = [""] * NUM_PLAYERS
            for player in data:
                idx = player["player_id"] - 1  # player_id is 1-based
                self.positions[idx, 0] = player["x"]
                self.positions[idx, 1] = player["y"]
                self.roles[idx] = player["role"]
            self.update_plot()
            print(f"Formation loaded from {file_path}")

    def mirror_formation(self):
        original_positions = np.copy(self.positions)
        mirrored_positions = np.copy(original_positions)
        mirrored_positions[:, 0] = 120 - original_positions[:, 0]  # Flip across length
        self.positions = mirrored_positions
        self.update_plot()


if __name__ == "__main__":
    root = tk.Tk()
    app = FormationEditor(root)
    root.mainloop()
