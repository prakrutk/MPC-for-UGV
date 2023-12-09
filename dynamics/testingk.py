from dynamics.carkinematics import Kinematics
import numpy as np

Nc = 3# Control Horizon
Np = 5# Prediction Horizon
initial_state = np.array([0,0,0,0,0,0,0,0]) # Initial state
x_i = np.array([0.0,0.0,0.0,0.0]) # x,y,theta,xdot,ydot,thetadot
u_i = np.array([0.0,0.0]) # v,omega
xr = np.array([0.0,0.0,0.0,0.0]) # Reference state
ur = np.array([0.0,0.0]) # Reference input
delu = 0.0*np.ones((2*Nc,1)) # Input rate of change
# Yreff = 0.0*np.ones((3*Np,1)) # Reference output
# Q = 100*np.identity(3*Np) # Weight matrix output
coeff = Kinematics(state = x_i
                ,input = u_i
                ,inputr = ur
                ,stater = xr
                ,delu = delu
                ,lf = 0.1
                ,lr = 0.1
                ,T = 0.1
                ,Nc = Nc
                ,Np = Np) 

for i in range (Nc):
    delu[2*i]=1.1
    delu[2*i+1]=0.1

Y,Y1,Y2,psi = coeff.Y(xr,ur,delu)
# print('Y1=',Y1)
# print('Y2=',Y2)
print('Y=',Y)
# print('psi=',psi)