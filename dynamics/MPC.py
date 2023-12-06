import numpy as np
from dynamics.cardynamics import dynamics
import cvxpy as cvx
from Pybullet.racecar_differential import pybullet_dynamics
import matplotlib as plt

Nc = 5 # Control Horizon
Np = 10 # Prediction Horizon
initial_state = np.array([0,0,0,0,0,0,0,0]) # Initial state
x_i = np.array([0.0,0.0,0.0,0.0,0.0,0.0]) # x,y,theta,xdot,ydot,thetadot
u_i = np.array([0.0,0.0]) # v,omega
xr = np.array([0.0,0.0,0.0,0.0,0.0,0.0]) # Reference state
ur = np.array([0.0,0.0]) # Reference input
delu = 0.0*np.ones((2*Nc,1)) # Input rate of change
Yreff = np.ones((3*Np,1)) # Reference output
Q = 100*np.identity(3*Np) # Weight matrix output 
R = 10*np.identity(2*Nc) # Weight matrix input
tolerance = 0.01*np.ones((3*Np,1)) # Tolerance
Ymax = Yreff + tolerance # Maximum output
Ymin = Yreff - tolerance # Minimum output
rho = 1 # Slack variable weight
epi = 0 # Slack variable
MAX_TIME = 100 # seconds

coeff = dynamics(state = x_i
                ,input = u_i
                ,inputr = ur
                ,stater = xr
                ,delu = delu
                ,cl = 0.7
                ,sf = 0.1
                ,cc = 8.2
                ,sr = 0.1
                ,m = 6.38
                ,alphaf = (5./360.)*2*np.pi
                ,lf = 0.1
                ,lr = 0.1
                ,iz = 0.058
                ,T = 1./240.
                ,Nc = Nc
                ,Np = Np) 

# Ynext = coeff.Y(x_i,u_i)

# Make a refernce state
def reff(i,x_i,midx,midy):
    x = x_i[0] + (i+1)*(midx - x_i[0])/Np
    y = x_i[1] + (i+1)*(midy - x_i[1])/Np
    theta = np.arctan2(y,x)
    xdot = 0
    ydot = 0
    thetadot = 0.0
    return x,y,theta

# Make a reference trajectory
def stateref(xr,x_i,midx,midy):
    x = x_i[0] + (midx - x_i[0])/Np
    y = x_i[1] + (midy - x_i[1])/Np
    theta = np.arctan2(y,x)
    xdot = 0
    ydot = 0
    thetadot = 0.0
    xr[0] = x
    xr[1] = y
    xr[2] = theta
    xr[3] = xdot
    xr[4] = ydot
    xr[5] = thetadot
    return xr

# Write a function to generate cubic spline between waypoints
# To be fixed
def spline(t,x_i,x,y):
    a = 2*x_i[0] - 2*x + y + x_i[1]
    b = -3*x_i[0] + 3*x - 2*y - x_i[1]
    c = x_i[1]
    d = x_i[0]
    x = a*t**3 + b*t**2 + c*t + d
    y = a*t**3 + b*t**2 + c*t + d
    theta = np.arctan2(y,x)
    return x,y

def nearest_index(x,y,x_i,y_i):
    dist = np.sqrt((x-x_i)**2 + (y-y_i)**2)
    return dist

# Solve the optimization problem
def linearmpc(x_i,u_i,xr,t,midx,midy):
    # dist = 0.0
    # for i in range(Np):
    #     distn = nearest_index(x_i[0],x_i[1],Yreff[3*i],Yreff[3*i+1])
    #     if distn < dist:
    #         dist = distn
    #         index = i
    # xr = np.concatenate(Yreff[3*index:3*index+2],np.array([0.0,0.0,0.0]))

    xr = stateref(xr,x_i,midx,midy)
    print('x_i[0] - midx = ',(x_i[0]-midx))
    print('x_i[1] - midy = ',(x_i[1]-midy))
    u = cvx.Variable((2*Nc +1,1))
    # u_t=u_t.reshape(2,1)
    cost = 0.0
    constraints = []
    for i in range(Np):
        # Yreff[3*i,0],Yreff[3*i+1,0],Yreff[3*i+2,0] = np.array(spline(i+1,x_i,x_i[0]+midx,x_i[1]+midy))
        Yreff[3*i,0],Yreff[3*i+1,0],Yreff[3*i+2,0] = np.array(reff(i+1,x_i,x_i[0]+midx,x_i[1]+midy))
    # print('Yreff=',Yreff)
    the_c=np.concatenate((coeff.theta(xr,ur),np.zeros((3*(Np-Nc),2*Nc))),axis=0)
    H = np.transpose(the_c).dot(Q).dot(the_c) + R 
    H = np.append(H,np.zeros((1,H.shape[1])),axis=0)
    c = np.zeros((H.shape[0],1))
    c[-1,0] = rho
    H = np.append(H,c,axis=1)
    E = coeff.phi(xr,ur).dot(np.concatenate((x_i-xr,u_i-ur),axis=0)).reshape(3*Np,1) - Yreff # Error term
    # print('E=',E)
    cost += cvx.quad_form(u,H) + 2*np.transpose(E).dot(Q).dot(the_c)@u[0:2*Nc,:] # Cost function
    for k in range(Nc):
        constraints += [u[2*k,:] <= 5.5] # Delu Input constraints
        constraints += [u[2*k,:] >= -1.5] # Delu Input constraints
        constraints += [u[2*k+1,:] <= 0.5] 
        constraints += [u[2*k+1,:] >= -0.5]
        constraints += [u_i[0] + u[2*k,:] <= 20]
        constraints += [u_i[1] + u[2*k+1,:] <= 1.5]
        constraints += [u_i[1] + u[2*k+1,:] >= -1.5]
        constraints += [u_i[0] + u[2*k,:] >= 1.5]
    constraints += [Ymin - u[2*Nc,:] <= coeff.Y(xr,ur)]
    constraints += [coeff.Y(xr,ur) <= Ymax + u[2*Nc,:]]
    
    prob = cvx.Problem(cvx.Minimize(cost), constraints) # Optimization problem initialization
    prob.solve()
    return u.value,u_i

# def check_waypoint(state,midx,midy):
#     if abs(state[0] - midx) < 0.05 and abs(state[1] - midy) <0.05:
#         return True
#     else:
#         return False

def check_goal(state, goal):
    if abs(state[0] - goal[0]) < 0.05 and abs(state[1] - goal[1]) < 0.05:
        return True
    else:
        return False

def simulate(initial_state,goal,cars,wheels,distance):
    goal = goal 
    state = initial_state
    time = 0.0
    u_t = np.array([0.0,0.0])
    x,phi,midx,midy = pybullet_dynamics.loop(0,200000,0,wheels,cars,distance)
    while MAX_TIME >= time:
        u, u_old = linearmpc(state,u_t,xr,time,midx,midy)
        for i in range(Nc):
            # x,phi = pyconnect(2*u[2*i,0],u[2*i+1,0],wheels,car,useRealTimeSim)
            x,phi,midx,midy = pybullet_dynamics.loop(u_old[0]+u[2*i,0],200000,u_old[1]+u[2*i+1,0],wheels,cars,distance)
            time += 1./240.
        state = np.array([(x[0]),(x[1]-20),phi[2],0.0,0.0,0.0])
        x_t = state
        u_t = np.array([u[2*Nc-2,0],u[2*Nc-1,0]])

        i +=1
        # if check_waypoint(state,midxn,midyn):
        #     midxn,midyn = midx,midy
        if check_goal(state, goal):
            break

def main():
    cars,wheels,distance = pybullet_dynamics.sim()
    initial_state = np.array([0,0,0,0,0,0])
    goal = np.array([-3,0,0,0,0,0])
    simulate(initial_state,goal,cars,wheels,distance)

if __name__ == '__main__':
    main()