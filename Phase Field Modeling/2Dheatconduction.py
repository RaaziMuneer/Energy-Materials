#2Dheatconduction

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

plt.rcParams["figure.figsize"] = (7, 6)
plt.rcParams["font.size"] = 12

# Physical parameters
alpha = 1.0e-4       # thermal diffusivity

# Grid
nx, ny = 80, 80
dx = dy = 1.0 / (nx - 1)

# Stability condition for explicit method
dt = 0.25 * dx**2 / alpha

# Simulation duration
n_steps = 500

print(f"dx = {dx:.5f}")
print(f"dt = {dt:.5e}")

# Temperature field
T = np.zeros((ny, nx))

# Hot spot in the center
cx, cy = nx // 2, ny // 2
radius = 8

for j in range(ny):
    for i in range(nx):
        if (i - cx)**2 + (j - cy)**2 < radius**2:
            T[j, i] = 100.0

# Boundary conditions
T[:, 0] = 0
T[:, -1] = 0
T[0, :] = 0
T[-1, :] = 0

plt.imshow(T, origin="lower")
plt.colorbar(label="Temperature")
plt.title("Initial distribution")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

def step_temperature(T, alpha, dt, dx, dy):
    Tn = T.copy()

    T[1:-1, 1:-1] = (
        Tn[1:-1, 1:-1]
        + alpha * dt * (
            (Tn[1:-1, 2:] - 2*Tn[1:-1, 1:-1] + Tn[1:-1, :-2]) / dx**2
            + (Tn[2:, 1:-1] - 2*Tn[1:-1, 1:-1] + Tn[:-2, 1:-1]) / dy**2
        )
    )

    # Dirichlet boundary conditions
    T[:, 0] = 0
    T[:, -1] = 0
    T[0, :] = 0
    T[-1, :] = 0

    return T

# Initialization
T_anim = T.copy()

fig, ax = plt.subplots()

im = ax.imshow(
    T_anim,
    origin="lower",
    cmap="hot",
    vmin=0,
    vmax=100,
    animated=True
)

cbar = plt.colorbar(im)
cbar.set_label("Temperature")

ax.set_title("2D Heat Conduction")
ax.set_xlabel("x")
ax.set_ylabel("y")

def update(frame):
    global T_anim

    for _ in range(3):
        T_anim = step_temperature(T_anim, alpha, dt, dx, dy)

    im.set_array(T_anim)
    ax.set_title(f"2D Heat Conduction – Schritt {frame*3}")

    return [im]

ani = FuncAnimation(
    fig,
    update,
    frames=150,
    interval=50,
    blit=True
)

plt.close(fig)

HTML(ani.to_jshtml())

# Further evolution to the final state
T_final = T.copy()

for _ in range(n_steps):
    T_final = step_temperature(T_final, alpha, dt, dx, dy)

plt.figure(figsize=(7,6))

plt.imshow(
    T_final,
    origin="lower",
    cmap="hot"
)

plt.colorbar(label="Temperature")

plt.title("Final state")
plt.xlabel("x")
plt.ylabel("y")

plt.show()
