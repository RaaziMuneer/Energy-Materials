# 1Dphasefieldmoving.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

gamma = 1.0
xi = 5.0
mu0 = 0.10
K = 1.0

Nx = 801
Lx = 400.0
x = np.linspace(0.0, Lx, Nx)
dx = x[1] - x[0]
dt = 0.01
nsteps = 20000
output_every = 200

def laplacian(phi):
    lap = np.zeros_like(phi)
    lap[1:-1] = (phi[2:] - 2.0 * phi[1:-1] + phi[:-2]) / dx**2
    return lap

def rhs(phi):
    lap = laplacian(phi)
    dw = phi * (1.0 - phi) * (1.0 - 2.0 * phi)
    dphi = K * lap - (4.0 * K / xi**2) * dw + (2.0 * K * mu0 / (gamma * xi)) * phi * (1.0 - phi)
    dphi[0] = 0.0
    dphi[-1] = 0.0
    return dphi

x0 = 0.75 * Lx
phi = 0.5 * (1.0 + np.tanh((x - x0) / xi))
phi[0] = 0.0
phi[-1] = 1.0

phi_history = [phi.copy()]
time_history = [0.0]

print("Running 1D Phase Field Moving Front simulation...")
for step in range(1, nsteps):
    phi[1:-1] += dt * rhs(phi)[1:-1]
    phi[0] = 0.0
    phi[-1] = 1.0

    if step % output_every == 0:
        phi_history.append(phi.copy())
        time_history.append(step * dt)

fig, ax = plt.subplots(figsize=(8, 4))
line, = ax.plot(x, phi_history[0])
ax.set_xlabel("x")
ax.set_ylabel("phi")
ax.set_ylim(-0.05, 1.05)
ax.grid(True)

def update(frame):
    line.set_ydata(phi_history[frame])
    ax.set_title(f"1D Moving Interface - Time = {time_history[frame]:.2f}")
    return line,

anim = FuncAnimation(
    fig, 
    update, 
    frames=len(phi_history), 
    interval=40, 
    blit=True, 
    cache_frame_data=False
)