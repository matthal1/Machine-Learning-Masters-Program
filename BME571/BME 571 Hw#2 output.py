import numpy as np
import matplotlib.pyplot as plt

# Define the given parameters
R_x = np.array([[1, 0.8182, 0.2021],
                [0.8182, 1, 0.3040],
                [0.2021, 0.3040, 1]])

d = np.array([0.8182, 0.3540, 0.2021])

# Initial weight vector
w_init = np.array([0.0, 0.0, 0.0])

# Learning rates
etas = [0.2, 1]
iterations = 50  # Number of iterations for gradient descent

# Store results for plotting
results = {}

for eta in etas:
    w = w_init.copy()
    trajectory = [w.copy()]
    energy = []
    
    for _ in range(iterations):
        grad_E = -d + np.dot(R_x, w)  # Compute gradient
        w = w - eta * grad_E  # Gradient descent update
        trajectory.append(w.copy())  # Store trajectory
        energy.append(0.5 - np.dot(d.T, w) + 0.5 * np.dot(w.T, np.dot(R_x, w)))  # Compute E(w)

    results[eta] = {
        "trajectory": np.array(trajectory),
        "energy": np.array(energy),
    }

# Create plots for both parts
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot weight trajectories (i)
for eta, res in results.items():
    traj = res["trajectory"]
    axes[0].plot(range(iterations + 1), traj[:, 0], label=f"w1, η = {eta}")
    axes[0].plot(range(iterations + 1), traj[:, 1], label=f"w2, η = {eta}", linestyle="dashed")
    axes[0].plot(range(iterations + 1), traj[:, 2], label=f"w3, η = {eta}", linestyle="dotted")

axes[0].set_xlabel("Iterations")
axes[0].set_ylabel("Weights (w1, w2, w3)")
axes[0].set_title("Gradient Descent Weight Updates")
axes[0].legend()
axes[0].grid()

# Plot energy function over iterations (ii)
for eta, res in results.items():
    axes[1].plot(range(iterations), res["energy"], label=f"η = {eta}")

axes[1].set_xlabel("Iterations")
axes[1].set_ylabel("E(w)")
axes[1].set_title("Energy Function Over Iterations")
axes[1].legend()
axes[1].grid()

plt.show()