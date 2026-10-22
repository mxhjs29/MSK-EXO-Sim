import numpy as np

def activation_step(a,u,times,dt):
    u=np.clip(u,0,1)
    factor=.5+1.5*np.clip(a,0,1)
    tau=np.where(u>a,times[:,0]*factor,times[:,1]/factor)
    return a+dt*(u-a)/np.maximum(tau,1e-8)

def activation_future(a,u,times,dt,steps=15):
    for _ in range(steps):a=activation_step(a,u,times,dt)
    return a

def excitation_for_target(a,target,times,dt):
    lo=np.zeros_like(a);hi=np.ones_like(a)
    for _ in range(24):
        mid=(lo+hi)/2
        below=activation_future(a,mid,times,dt)<target
        lo=np.where(below,mid,lo);hi=np.where(below,hi,mid)
    return (lo+hi)/2
