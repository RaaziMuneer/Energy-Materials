# 1Dphasefieldheatconduction.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

gamma = 1.0
xi = 5.0
K = 1.0
alpha = 0.1
Lh = 1.0
c = 1.0
TM = 1.0
Tinfty = 0.8

Nx = 101
Lx = 200.0
x = np.linspace(0, Lx, Nx)
dx = x[1] - x[0]
dt = 0.1
nsteps = 10000
output_every = 100

def laplacian(f):
    lap = np.zeros_like(f)
    lap[1:-1] = (f[2:] - 2*f[1:-1] + f[:-2]) / dx**2
    return lap

def dh(phi):
    return 6*phi*(1-phi)

x0 = 0.75 * Lx
phi = 0.5 * (1 + np.tanh((x - x0) / xi))
phi[0] = 1.0
phi[-1] = 0.0

T = np.full_like(x, TM)
T[x < x0] = Tinfty
T[0] = Tinfty
T[-1] = TM

# Precompute simulation history
phi_history = [phi.copy()]
T_history = [T.copy()]
time_history = [0.0]

print("Running 1D Phase Field Heat Conduction simulation...")
for step in range(1, nsteps):
    lap_phi = laplacian(phi)
    dphi_dt = (
        K * lap_phi
        - (4 * K / xi**2) * phi * (1 - phi) * (1 - 2 * phi)
        - (2 * K * Lh / (gamma * xi * TM)) * (T - TM) * phi * (1 - phi)
    )

    phi[1:-1] += dt * dphi_dt[1:-1]
    phi[0] = 0.0
    phi[-1] = 1.0

    lap_T = laplacian(T)
    dT_dt = alpha * lap_T + (Lh / c) * dh(phi) * dphi_dt
    T[1:-1] += dt * dT_dt[1:-1]
    T[0] = Tinfty
    T[-1] = TM

    if step % output_every == 0:
        phi_history.append(phi.copy())
        T_history.append(T.copy())
        time_history.append(step * dt)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
line_phi, = ax[0].plot(x, phi_history[0])
ax[0].set_ylim(-0.05, 1.05)
ax[0].set_title("Phase field")

line_T, = ax[1].plot(x, T_history[0])
ax[1].set_title("Temperature")

def update(frame):
    line_phi.set_ydata(phi_history[frame])
    line_T.set_ydata(T_history[frame])
    fig.suptitle(f"Time = {time_history[frame]:.2f}")
    return line_phi, line_T

anim = FuncAnimation(
    fig, 
    update, 
    frames=len(phi_history), 
    interval=50, 
    blit=False, 
    cache_frame_data=False
)