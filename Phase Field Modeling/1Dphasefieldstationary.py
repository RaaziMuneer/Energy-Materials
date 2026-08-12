# 1Dphasefieldstationary.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

gamma = 1.0
xi = 5.0
mu0 = 0.0
K = 1.0

Nx = 401
Lx = 200.0
x = np.linspace(0.0, Lx, Nx)
dx = x[1] - x[0]
dt = 0.01
nsteps = 15000
output_every = 150

def laplacian(phi):
    lap = np.zeros_like(phi)
    lap[1:-1] = (phi[2:] - 2 * phi[1:-1] + phi[:-2]) / dx**2
    return lap

def rhs(phi):
    lap = laplacian(phi)
    dw = phi * (1 - phi) * (1 - 2 * phi)
    dphi = K * lap - (4 * K / xi**2) * dw + (2 * K * mu0 / (gamma * xi)) * phi * (1 - phi)
    dphi[0] = 0.0
    dphi[-1] = 0.0
    return dphi

x0 = Lx / 2
phi = np.zeros_like(x)
phi[x > x0] = 1.0
phi[0] = 0.0
phi[-1] = 1.0

phi_history = [phi.copy()]

print("Running 1D Phase Field Stationary Relaxation simulation...")
for step in range(1, nsteps):
    phi[1:-1] += dt * rhs(phi)[1:-1]
    phi[0] = 0.0
    phi[-1] = 1.0

    if step % output_every == 0:
        phi_history.append(phi.copy())

fig, ax = plt.subplots(figsize=(8, 4))
line, = ax.plot(x, phi_history[0])
ax.set_xlabel("x")
ax.set_ylabel("phi")
ax.set_ylim(-0.05, 1.05)
ax.grid(True)

def update(frame):
    line.set_ydata(phi_history[frame])
    ax.set_title(f"1D Stationary Relaxation Step {frame * output_every}")
    return line,

anim = FuncAnimation(
    fig, 
    update, 
    frames=len(phi_history), 
    interval=40, 
    blit=True, 
    cache_frame_data=False
)