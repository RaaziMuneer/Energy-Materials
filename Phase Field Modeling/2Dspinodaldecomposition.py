#2Dspinodaldecomposition

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

plt.rcParams["figure.figsize"] = (7, 7)
plt.rcParams["font.size"] = 12

# Grid size
nx, ny = 128, 128

# Physical dimensions
dx = dy = 1.0

# Time step
dt = 0.01

# Model parameters
M = 1.0
kappa = 1.0

# Number of simulation steps
n_steps = 3000

# Visualization interval
plot_every = 10

# Random initial fluctuations around c = 0
np.random.seed(42)

c = 0.02 * (2 * np.random.rand(ny, nx) - 1)

plt.imshow(c, cmap="coolwarm", origin="lower")
plt.colorbar(label="Concentration")
plt.title("Initial Concentration Field")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

def laplacian(field):
    return (
        np.roll(field, +1, axis=0)
        + np.roll(field, -1, axis=0)
        + np.roll(field, +1, axis=1)
        + np.roll(field, -1, axis=1)
        - 4 * field
    ) / dx**2

def chemical_potential(c):
    # Derivative of the double-well potential
    dfdc = c**3 - c

    # Chemical potential
    mu = dfdc - kappa * laplacian(c)

    return mu

def step_cahn_hilliard(c):
    mu = chemical_potential(c)

    # Cahn–Hilliard evolution
    c_new = c + dt * M * laplacian(mu)

    return c_new

c_anim = c.copy()

fig, ax = plt.subplots()

im = ax.imshow(
    c_anim,
    cmap="coolwarm",
    origin="lower",
    vmin=-1,
    vmax=1,
    animated=True
)

cbar = plt.colorbar(im)
cbar.set_label("Concentration")

ax.set_title("Spinodal Decomposition")
ax.set_xlabel("x")
ax.set_ylabel("y")

def update(frame):
    global c_anim

    for _ in range(plot_every):
        c_anim = step_cahn_hilliard(c_anim)

    im.set_array(c_anim)

    ax.set_title(f"Spinodal Decomposition – Step {frame * plot_every}")

    return [im]

ani = FuncAnimation(
    fig,
    update,
    frames=n_steps // plot_every,
    interval=40,
    blit=True,
    cache_frame_data=False  
)

plt.close(fig)

c_final = c.copy()

for _ in range(n_steps):
    c_final = step_cahn_hilliard(c_final)

plt.figure(figsize=(7,7))

plt.imshow(
    c_final,
    cmap="coolwarm",
    origin="lower",
    vmin=-1,
    vmax=1
)

plt.colorbar(label="Concentration")

plt.title("Final Concentration Field")
plt.xlabel("x")
plt.ylabel("y")

plt.show()

def free_energy(c):
    grad_x = (np.roll(c, -1, axis=1) - c) / dx
    grad_y = (np.roll(c, -1, axis=0) - c) / dy

    bulk = 0.25 * (c**2 - 1)**2
    gradient = 0.5 * kappa * (grad_x**2 + grad_y**2)

    return np.sum(bulk + gradient)

c_energy = c.copy()

energies = []

for step in range(1000):
    c_energy = step_cahn_hilliard(c_energy)

    if step % 5 == 0:
        energies.append(free_energy(c_energy))

plt.figure(figsize=(7,5))

plt.plot(energies)

plt.xlabel("Saved Step")
plt.ylabel("Free Energy")
plt.title("Free Energy Evolution")

plt.grid(True)

plt.show()
