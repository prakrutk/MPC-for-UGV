from dynamics.cardynamicsupd import dynamics
import numpy as np
import matplotlib.pyplot as plt

# Define the car model
Nc = 3# Control Horizon
Np = 5# Prediction Horizon
initial_state = np.array([0,0,0,0]) # Initial state
x_i = np.array([0.0,0.0,0.0,0.0]) # x,y,theta,xdot,ydot,thetadot
u_i = np.array([0.0]) # v,omega
xr = np.array([0.0,0.0,0.0,1.0]) # Reference state
ur = np.array([0.0]) # Reference input
delu = 0.0*np.ones((Nc,1)) # Input rate of change
Yreff = 0.0*np.ones((3*Np,1)) # Reference output
Q = 100*np.identity(3*Np) # Weight matrix output
V = 10
coeff = dynamics(state = x_i
                ,input = u_i
                ,inputr = ur
                ,stater = xr
                ,delu = delu
                ,cl = 0.7
                ,sf = 0.0
                ,cc = 8.2
                ,sr = 0.0
                ,m = 6.38
                ,alphaf = 0
                ,lf = 0.1
                ,lr = 0.1
                ,iz = 0.058
                ,T = 1./240.
                ,Nc = Nc
                ,Np = Np
                ,V = V) 

for i in range (Nc):
    delu[i]=0.1

Y,Y1,Y2 = coeff.Y((x_i-xr),u_i,delu)
# print('Y1=',Y1)
# print('Y2=',Y2)
print('Y=',Y)