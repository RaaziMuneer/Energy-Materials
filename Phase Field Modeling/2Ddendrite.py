#2Ddendrite
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
nx=ny=300
Lx=Ly=9.0
dx=Lx/nx
dy=Ly/ny
dt=2e-4
eps0=0.01
tau=3e-4
alpha=0.9
gamma=10.0
latent=2.0
delta=0.02
jmode=4
theta0=0.0
noise_amp=0.01
p=np.zeros((ny,nx))
T=np.zeros((ny,nx))
cx=nx//2
cy=ny//2
r0=5
Y,X=np.ogrid[:ny,:nx]
p[(X-cx)**2+(Y-cy)**2<r0**2]=1.0
def laplacian(u):
    return (np.roll(u,1,0)+np.roll(u,-1,0)+np.roll(u,1,1)+np.roll(u,-1,1)-4*u)/(dx*dx)

def m_of_T(T):
    return alpha/np.pi*np.arctan(gamma*(1.0-T))

def anisotropy(p):
    px=(np.roll(p,-1,1)-np.roll(p,1,1))/(2*dx)
    py=(np.roll(p,-1,0)-np.roll(p,1,0))/(2*dy)
    theta=np.arctan2(py,px)
    eps=eps0*(1+delta*np.cos(jmode*(theta-theta0)))
    deps=-eps0*delta*jmode*np.sin(jmode*(theta-theta0))
    return eps,deps,px,py
def step(p,T):
    eps,deps,px,py=anisotropy(p)
    term1=-(np.roll(eps*deps*py,-1,1)-np.roll(eps*deps*py,1,1))/(2*dx)
    term2=(np.roll(eps*deps*px,-1,0)-np.roll(eps*deps*px,1,0))/(2*dy)
    fx=eps**2*px
    fy=eps**2*py
    div=(np.roll(fx,-1,1)-np.roll(fx,1,1))/(2*dx)+(np.roll(fy,-1,0)-np.roll(fy,1,0))/(2*dy)
    rhs=term1+term2+div+p*(1-p)*(p-0.5+m_of_T(T))
    noise=noise_amp*p*(1-p)*(np.random.random(p.shape)-0.5)
    dpdt=(rhs+noise)/tau
    p=p+dt*dpdt
    p=np.clip(p,0,1)
    T=T+dt*(laplacian(T)+latent*dpdt)
    return p,T
frames=[]
for n in range(2000):
    p,T=step(p,T)
    if n%20==0:
        frames.append(p.copy())
print('stored',len(frames),'frames')
fig,ax=plt.subplots(figsize=(6,6))
ax.imshow(frames[-1],origin='lower',cmap='gray')
ax.set_title('Final phase field')
plt.show()
fig,ax=plt.subplots(figsize=(6,6))
im=ax.imshow(frames[0],origin='lower',cmap='gray',animated=True)
def update(i):
    im.set_array(frames[i])
    return [im]
ani=FuncAnimation(fig,update,frames=len(frames),interval=50)
HTML(ani.to_jshtml())
