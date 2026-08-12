#1D heat conduction

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# Improve plot appearance
plt.rcParams["figure.figsize"] = (10, 5)
plt.rcParams["font.size"] = 12

# Physical parameters
length = 1.0          # Length of the rod [m]
alpha = 0.01          # Thermal diffusivity

# Numerical parameters
nx = 100              # Number of spatial points
dx = length / (nx - 1)

dt = 0.0004           # Time step
nt = 800              # Number of time steps

# Stability condition for explicit scheme
stability = alpha * dt / dx**2
print(f"Stability parameter = {stability:.4f}")

if stability > 0.5:
    print("WARNING: Explicit scheme may be unstable!")

# Spatial grid
x = np.linspace(0, length, nx)

# Initial temperature distribution
T = np.zeros(nx)

# Heat pulse in the center
T[int(nx * 0.4):int(nx * 0.6)] = 100

# Boundary conditions
T_left = 0
T_right = 0

# Store solution history
history = [T.copy()]

for n in range(nt):
    T_new = T.copy()

    # Finite difference update
    for i in range(1, nx - 1):
        T_new[i] = (
            T[i]
            + alpha * dt / dx**2
            * (T[i+1] - 2*T[i] + T[i-1])
        )

    # Apply boundary conditions
    T_new[0] = T_left
    T_new[-1] = T_right

    T = T_new
    history.append(T.copy())

history = np.array(history)

print("Simulation completed.")
print("History shape:", history.shape)


times_to_plot = [0, 50, 150, 300, 600, 800]

plt.figure()

for t in times_to_plot:
    plt.plot(x, history[t], label=f"Step {t}")

plt.xlabel("Position along rod")
plt.ylabel("Temperature")
plt.title("1D Heat Conduction")
plt.legend()
plt.grid(True)

plt.show()

fig, ax = plt.subplots()

line, = ax.plot(x, history[0], lw=2)

ax.set_xlim(0, length)
ax.set_ylim(0, np.max(history) * 1.1)

ax.set_xlabel("Position along rod")
ax.set_ylabel("Temperature")
ax.set_title("Heat Diffusion Animation")
ax.grid(True)

def update(frame):
    line.set_ydata(history[frame])
    ax.set_title(f"Heat Diffusion Animation - Step {frame}")
    return line,

anim = FuncAnimation(
    fig,
    update,
    frames=len(history),
    interval=20,
    blit=True
)

HTML(anim.to_jshtml())
