#2Dphasefieldnucleus
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
gamma = 1.0
xi    = 4.0
K     = 1.0
mu0   = 0.1

Rc_theory = gamma / mu0
print(f"Theoretical critical radius Rc = {Rc_theory:.3f}")
Nx = 256
Ny = 256
dx = 1.0
dt = 0.02
nsteps = 6000
save_every = 20
R0 = 30.0
x = np.arange(Nx)*dx
y = np.arange(Ny)*dx
X, Y = np.meshgrid(x, y, indexing="ij")

xc = Nx*dx/2
yc = Ny*dx/2

r = np.sqrt((X-xc)**2 + (Y-yc)**2)
phi = 0.5*(1.0 - np.tanh((r-R0)/(xi/2)))
def laplacian(field):
    return (
        np.roll(field,+1,axis=0)
      + np.roll(field,-1,axis=0)
      + np.roll(field,+1,axis=1)
      + np.roll(field,-1,axis=1)
      - 4.0*field
    ) / dx**2

def rhs(phi):
    lap = laplacian(phi)

    double_well = (
        -(4.0*K/xi**2)
        * phi*(1.0-phi)*(1.0-2.0*phi)
    )

    driving = (
        (2.0*K*mu0/(gamma*xi))
        * phi*(1.0-phi)
    )

    return K*lap + double_well + driving

def effective_radius(phi):
    area = np.sum(phi)*dx*dx
    return np.sqrt(area/np.pi)
snapshots = []
times = []
radius_history = []

for step in range(nsteps):

    phi += dt*rhs(phi)
    phi = np.clip(phi,0.0,1.0)

    radius_history.append(effective_radius(phi))

    if step % save_every == 0:
        snapshots.append(phi.copy())
        times.append(step*dt)

    if step % 500 == 0:
        print(step)
time_radius = np.arange(nsteps)*dt

plt.figure(figsize=(8,5))
plt.plot(time_radius, radius_history, lw=2, label="Phase-field")
plt.axhline(Rc_theory, ls="--", label=f"Rc={Rc_theory:.2f}")
plt.xlabel("time")
plt.ylabel("effective radius")
plt.grid()
plt.legend()
plt.show()
R_theory = [R0]

for i in range(nsteps-1):
    R = R_theory[-1]
    dRdt = K*mu0/gamma - K/R
    Rnew = R + dt*dRdt
    R_theory.append(max(Rnew,0))
plt.figure(figsize=(8,5))
plt.plot(time_radius, radius_history, lw=2, label="Phase-field")
plt.plot(time_radius, R_theory, "--", lw=2, label="Sharp-interface theory")
plt.xlabel("time")
plt.ylabel("radius")
plt.grid()
plt.legend()
plt.show()
plt.figure(figsize=(6,6))
plt.imshow(phi.T, origin="lower", cmap="viridis")
plt.colorbar(label="phi")
plt.title("Final phase field")
plt.tight_layout()
plt.show()
fig, ax = plt.subplots(figsize=(6,6))

im = ax.imshow(
    snapshots[0].T,
    origin="lower",
    cmap="viridis",
    animated=True,
    vmin=0,
    vmax=1
)

plt.colorbar(im)
title = ax.set_title("")

def update(frame):
    im.set_array(snapshots[frame].T)
    title.set_text(f"t = {times[frame]:.2f}")
    return im, title

ani = FuncAnimation(
    fig,
    update,
    frames=len(snapshots),
    interval=50,
    blit=False
)

plt.close(fig)
HTML(ani.to_jshtml())
