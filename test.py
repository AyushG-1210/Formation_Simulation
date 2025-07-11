import numpy as np

X = np.load("C:\\Users\\anshu\\Desktop\\Simulation_Project\\Working_Data\\X_players.npy", allow_pickle=True)
print(f"Shape X = {X.shape}")
x = np.load("C:\\Users\\anshu\\Desktop\\Simulation_Project\\Working_Data\\X_ball.npy", allow_pickle=True)
print(f"Shape ball = {x.shape}")
y = np.load("C:\\Users\\anshu\\Desktop\\Simulation_Project\\Working_Data\\y_players.npy", allow_pickle=True)
print(f"Shape y = {y.shape}")
# pred = np.load("C:\\Users\\anshu\\Desktop\\Simulation_Project\\ML_Part\\model_predictions_with_ball.npy", allow_pickle=True)
# print(f"Shape preds = {pred.shape}")
# print(f"Top 5 -> {pred[0][:5]}")
t = np.load("C:\\Users\\anshu\\Desktop\\Simulation_Project\\Working_Data\\test_predictions.npy", allow_pickle=True)
print(f"Shape test = {t.shape}")
