import numpy as np
from dynamics.cardynamics import dynamics
import cvxpy as cvx
from Pybullet.racecar_differential import pybullet_dynamics
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from math import sin, cos, tan, atan2, sqrt, pi
from dynamics.carkinematics import Kinematics

Nc = 5# Control Horizon
Np = 10# Prediction Horizon
initial_state = np.array([0,0,0,0,0,0,0,0]) # Initial state
x_i = np.array([0.0,0.0,0.0,0.0,0.0,0.0]) # x,y,theta,xdot,ydot,thetadot
# initial_state = np.array([0,0,0,0,0,0]) # Initial state
# x_i = np.array([0.0,0.0,0.0,0.0]) # x,y,theta,xdot,ydot,thetadot
u_i = np.array([0.0,0.0]) # v,omega
# xr = np.array([0.0,0.0,0.0,0.0]) # Reference state
xr = np.array([0.0,0.0,0.0,0.0,0.0,0.0]) # Reference state
ur = np.array([0.0,0.0]) # Reference input
delu = 0.0*np.ones((2*Nc,1)) # Input rate of change
Yreff = 0.0*np.ones((3*Np,1)) # Reference output
Q = 100*np.identity(3) # Weight matrix output
# R = np.zeros((2*Nc,2*Nc)) # Weight matrix input
#diag = np.array([0.1,100])
#for i in range(Nc-1): 
    #diag = np.append(diag,diag)
#R = np.diagonal(diag) 
R = np.zeros((2*Nc,2*Nc))
for i in range(Nc):
    R[2*i,2*i+0]=0.1
    R[2*i,2*i+1]=0
    R[2*i+1,2*i+0]=0
    R[2*i+1,2*i+1]=100# Weight matrix input
# print(R)
R = np.array([[0.1,0.0],[0.0,100]])
tolerance = 0.01*np.ones((3*Np,1)) # Tolerance
Ymax = Yreff + tolerance # Maximum output
Ymin = Yreff - tolerance # Minimum output
rho = 10 # Slack variable weight
epi = 1 # Slack variable
MAX_TIME = 100 # seconds

# Call the dynamics class
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
                ,alphaf = (5./360.)*2*np.pi
                ,lf = 0.1
                ,lr = 0.05
                ,iz = 0.058
                ,T = 1./240.
                ,Nc = Nc
                ,Np = Np) 

kine = Kinematics(state = x_i,
                input = u_i,
                inputr = ur,
                stater = xr,
                delu = delu,
                lf = 0.1,
                lr = 0.1,
                T = 1./240.,
                Nc = Nc,
                Np = Np)


# Ynext = coeff.Y(x_i,u_i)

def cubicspline(xr,x_i,midx,midy):
    x1=xr[0]
    y1=xr[1]
    x2=x_i[0]+midx
    y2=x_i[1]+midy
    # x3=x_i[0]+midx[1]
    # y3=x_i[1]+midy[1]
    if x1 < x2 :
            x = np.array([x1,x2])
            y = np.array([y1,y2])
            r = 0
    else :
            x = np.array([x2,x1])
            y = np.array([y2,y1])
            r = 1
    Yr = np.zeros((3*Np,1))
    Yre = np.zeros((3,1))   
    Yf = np.zeros((3*Np,1))
    for t in range(Np):    
        
        # y = np.array([y1,y2])
        x_interp = np.linspace(np.min(x),np.max(x),Np)
        x_p = x_interp[t]
        y_cubicBC = CubicSpline(x,y, bc_type="natural")
        y_p = y_cubicBC(x_p)
        if t<Np-1:
            x_p1 = x_interp[t]
            y_p1= y_cubicBC(x_p1)
        else:
            x_p1 = x_interp[t-1]
            y_p1= y_cubicBC(x_p1)
        if t<Np-1:
            x_p2 = x_interp[t+1]
            y_p2= y_cubicBC(x_p2)
        else:
            x_p2 = x_interp[t]
            y_p2= y_cubicBC(x_p2)
        theta_p=np.arctan(((y_p2-y_p1)/(x_p2-x_p1)))
        Yre[0] = x_p
        Yre[1] = y_p 
        Yre[2] = theta_p
        Yr[3*t:3*t+3] = Yre
        Yf[3*t:3*t+3]=np.flip(Yre)
        #y_p = y_cubicBC(x_p1)
        # ytemp = np.zeros(y_p.shape[0],y_p.shape[1])
        # xtemp = np.zeros(x_p.shape[0],x_p.shape[1])

    if r == 1:
        Yr = np.flip(Yf) 
    return Yr

def trajectory(i,x_i):
    x = x_i[0] + 0.1*i
    y = 0
    # y = x_i[1] + 0.1*i
    theta = 0
    xdot = 0
    ydot = 0
    thetadot = 0.0
    return x,y,theta

def stater(i,x_i):
    x = x_i[0] + 0.1*i
    y = 0
    # y = x_i[1] + 0.1*i
    theta = 0
    xdot = 0
    ydot = 0
    thetadot = 0.0
    return x,y,theta,xdot,ydot,thetadot

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
def linearmpc(x_i,u_i,xr,t,midx,midy,Yreff):
    dist = 100.0
    for i in range(Np):
        distn = nearest_index(x_i[0],x_i[1],Yreff[3*i],Yreff[3*i+1])
        if distn < dist:
            dist = distn
            index = i
    # xr = np.concatenate(Yreff[3*index:3*index+2],np.array([0.0,0.0,0.0]))
    xr[0] = Yreff[3*index]
    # print('xr[0]=',xr[0])
    # print('midx_g=',x_i[0]+midx)
    # print('midy_g=',x_i[1]+midy)
    xr[1] = Yreff[3*index+1]
    xr[2] = Yreff[3*index+2]
    xr[3] = (Yreff[3*index+3] - Yreff[3*index])*240
    xr[4] = (Yreff[3*index+4] - Yreff[3*index+1])*240
    xr[5] = (Yreff[3*index+5] - Yreff[3*index+2])*240

    ur[0] = xr[3]  
    # print(xr)
    # print('x_i=',x_i)
    # print('x_i + midx = ',(x_i[0]+midx))
    # print('x_i + midy = ',(x_i[1]+midy))
    # xr = stateref(xr,x_i,(x_i[0]+midx),(x_i[1]+midy))
    # xr = np.array(stater(0,x_i))
    # print('xr=',xr)
    # print('x_i[0] - midx = ',(x_i[0]-midx))
    # print('x_i[1] - midy = ',(x_i[1]-midy))
    u = cvx.Variable((2*Nc+1,1))
    Y = cvx.Parameter((3*Np,1))
    usc = 0.0
    uvc = 0.0
    # Y = cvx.Variable((3*Np,1))
    # E = cvx.Parameter((3*Np,1))
    # u_t=u_t.reshape(2,1)
    cost = 0.0
    constraints = []
    # print(midx)
    if t <= 100/240:
        for i in range(Np):
        # Yreff[3*i,0],Yreff[3*i+1,0],Yreff[3*i+2,0] = np.array(spline(i+1,x_i,x_i[0]+midx,x_i[1]+midy))
        # Yreff[3*i,0],Yreff[3*i+1,0],Yreff[3*i+2,0] = np.array(reff(i+1,x_i,x_i[0]+midx,x_i[1]+midy))
        # Yreff[3*i,0],Yreff[3*i+1,0],Yreff[3*i+2,0] = np.array(cubicspline(i,xr[0],xr[1],x_i[0]+midx,x_i[1]+midy))
    # Yreff = cubicspline(xr,x_i,midx,midy)
    # print('Yreff=',Yreff)
            Yreff[3*i,0],Yreff[3*i+1,0],Yreff[3*i+2,0] = np.array(trajectory(i,xr))
    # print('Yreff=',Yreff)
    # the_c=np.concatenate((kine.theta(xr,ur),np.zeros((3*(Np-Nc),2*Nc))),axis=0)
    # H = np.transpose(the_c).dot(Q).dot(the_c) + R 
    # H = np.append(H,np.zeros((1,H.shape[1])),axis=0)
    # c = np.zeros((H.shape[0],1))
    # c[-1,0] = rho
    # H = np.append(H,c,axis=1)
    the_c=np.concatenate((coeff.theta(xr,ur),np.zeros((3*(Np-Nc),2*Nc))),axis=0)
    H = np.transpose(the_c).dot(Q).dot(the_c) + R 
    H = np.append(H,np.zeros((1,H.shape[1])),axis=0)
    c = np.zeros((H.shape[0],1))
    c[-1,0] = rho
    H = np.append(H,c,axis=1) 
    stated = x_i -xr
    inputd = u_i -ur
    x_i_g = x_i[0:3]
    for i in range(Np-1):
        x_i_g = np.append(x_i_g,x_i[3*(i+1):3*(i+1)+3],axis = 0)
    Y= np.array(coeff.phi(xr,ur).dot(np.concatenate((stated,inputd),axis=0)).reshape((3*Np,1))) + the_c@u[0:2*Nc,:]
    # Y = np.array(kine.phi(xr,ur).dot(np.concatenate((x_i_g,u_i),axis=0)).reshape((3*Np,1))) + the_c@u[0:2*Nc,:] + kine.psi(xr,ur)
    # E = kine.phi(xr,ur).dot(np.concatenate((x_i_g,u_i),axis=0)).reshape((3*Np,1)) + kine.psi(xr,ur) - Yreff
    E = coeff.phi(xr,ur).dot(np.concatenate((x_i,u_i),axis=0)).reshape(3*Np,1) - Yreff # Error term
    # E = Y - the_c@u[0:2*Nc,:] - Yreff
    Ymin = Yreff - tolerance
    #print(Ymin.shape)
    Ymax = Yreff + tolerance
    # Y= np.array(coeff.phi(xr,ur).dot(np.concatenate((stated,inputd),axis=0)).reshape((3*Np,1))) + the_c@u[0:2*Nc,:]
    # print('E=',E)
    # cost += cvx.quad_form(Y-Yreff,Q) + cvx.quad_form(u,R) # Cost function
    # cost+= cvx.quad_form(Y-Yreff,Q) + cvx.quad_form(u[0:2*Nc,:],R) + rho*cvx.norm(u[2*Nc,:],1) # Cost function
    cost += cvx.quad_form(u,H) + np.transpose(E).dot(Q).dot(the_c)@u[0:2*Nc,:] # Cost functionsetJointMotorControl2forces
    # cost += cvx.quad_form(u,H)
    for k in range(Nc):
        constraints += [u[2*k,:] <= 0.5] # Delu Input constraints
        constraints += [u[2*k,:] >= -0.5] # Delu Input constraints
        constraints += [u[2*k+1,:] <= 0.05] 
        constraints += [u[2*k+1,:] >= -0.05]
        uvc += u[2*k,:]
        usc += u[2*k+1,:]
        # constraints += [u_i[0] + u[2*k,:] <= 10]
        # constraints += [u_i[1] + u[2*k+1,:] <= 0.5]
        # constraints += [u_i[1] + u[2*k+1,:] >= -0.5]
        # constraints += [u_i[0] + u[2*k,:] >= -10]
    constraints += [u_i[0] + uvc <= 10]
    constraints += [u_i[1] + usc <= 0.5]
    constraints += [u_i[1] + usc >= -0.5]
    constraints += [u_i[0] + uvc >= -10]
    # constraints += [u[2*Nc,:] >= -epi]
    # constraints += [u[2*Nc,:] <= epi]
    #ep= ep.reshape(75,1)
    #print(ep.shape)
    #constraints += [Ymin - np.ones([3*Np,1])@u[2*Nc,:] <= Y]
    for i in range(3*Np):
        constraints += [Ymin[i,0] - u[2*Nc,:] <= Y[i,0]]
        constraints += [Y[i,0] <= Ymax[i,0] + u[2*Nc,:]]
    #constraints += [Ymin - ep <= Y]
    #constraints += [Y <= Ymax + np.ones([3*Np,1])@u[2*Nc,:]]
    #constraints += [Y <= Ymax + ep]
    
    prob = cvx.Problem(cvx.Minimize(cost), constraints) # Optimization problem initialization
    # prob.solve(solver=cvx.ECOS,verbose=False) # Solver
    prob.solve()
    # print()
    # print(Y.value-Yreff)
    # print('status=',prob.status)
    # print('cost=',cost.value)
    #print('del_u=',u.value)
    return u.value,u_i,Yreff

def MPC(x_i,u_i,xr,t,midx,midy,Yreff):
    dist = 100.0
    for i in range(Np):
        distn = nearest_index(x_i[0],x_i[1],Yreff[3*i],Yreff[3*i+1])
        if distn < dist:
            dist = distn
            index = i
    # xr = np.concatenate(Yreff[3*index:3*index+2],np.array([0.0,0.0,0.0]))

    ur[0] = xr[3]  
    u = cvx.Variable((2,1))
    x_next = cvx.Variable((6,1))
    # Y = cvx.Variable((3*Np,1))
    usc = 0.0
    uvc = 0.0
    x_next = x_i
    cost = 0.0
    constraints = []
    for i in range(Np):
        xr[0] = Yreff[3*i]
        xr[1] = Yreff[3*i+1]
        xr[2] = Yreff[3*i+2]
        xr[3] = (Yreff[3*i+3] - Yreff[3*i])*240
        xr[4] = (Yreff[3*i+4] - Yreff[3*i+1])*240
        xr[5] = (Yreff[3*i+5] - Yreff[3*i+2])*240
        xr = xr.reshape(6,1)
        x_next = coeff.eqn(x_next,u)
        cost += cvx.quad_form(x_next-xr,Q) + cvx.quad_form(u,R)
        constraints += [u[2*i,:] <= 0.5]
        constraints += [u[2*i,:] >= -0.5]
        constraints += [u[2*i+1,:] <= 0.05]
        constraints += [u[2*i+1,:] >= -0.05]
    
    prob = cvx.Problem(cvx.Minimize(cost), constraints) # Optimization problem initialization
    prob.solve(solver=cvx.ECOS,verbose=False) # Solver
    return u.value,u_i,Yreff



    # Y = cvx.Variable((3*Np,1))
    # E = cvx.Parameter((3*N

# def check_waypoint(state,midx,midy):
#     if abs(state[0] - midx) < 0.05 and abs(state[1] - midy) <0.05:
#         return True
#     else:
#         return False

# Check goal location
def check_goal(state, goal):
    if abs(state[0] - goal[0]) < 0.05 and abs(state[1] - goal[1]) < 0.05:
        return True
    else:
        return False

def simulate(initial_state,goal,cars,wheels,distance,Yreff):
    goal = goal 
    state = initial_state
    time = 0.0
    # j = 0
    u_t = np.array([0.0,0.0])
    x,phi,midx,midy,vel,omega = pybullet_dynamics.loop(0,10,0,wheels,cars,distance,Yreff)
    while MAX_TIME >= time:
        # u, u_old,Yreff = linearmpc(state,u_t,xr,time,midx,midy,Yreff)
        u, u_old,Yreff = MPC(state,u_t,xr,time,midx,midy,Yreff)
        for i in range(Nc):
            # x,phi = pyconnect(2*u[2*i,0],u[2*i+1,0],wheels,car,useRealTimeSim)
            # print('u[2*i,0] + u_old[0]=',u[2*i,0]+u_old[0])
            # print('u[2*i+1,0] + u_old[1]=',u[2*i+1,0]+u_old[1])
            x,phi,midx,midy,vel,omega = pybullet_dynamics.loop(u_old[0]+u[2*i,0],10,u_old[1]+u[2*i+1,0],wheels,cars,distance,Yreff)
            u_old[0] = u_old[0]+u[2*i,0]
            u_old[1] = u_old[1]+u[2*i+1,0]
            time += 1./240.
        # velo = np.sqrt(vel[0]**2 + vel[1]**2)
        state = np.array([(x[0]),(x[1] - 20),phi[2],vel[0],vel[1],omega[2]])
        # state = np.array([(x[0]),(x[1]),velo,phi[2]])
        # print('state=',state)
        # j+=1
        # print('corresponding Yreff=',Yreff[3*Nc:3*Nc+3])
        # plt.cla()
        # plt.plot(state[0],state[1],'ro')
        # plt.plot(Yreff[3*Nc],Yreff[3*Nc+1],'bo')
        # plt.axis("equal")
        # plt.grid(True)
        # plt.pause(0.0001)
        # print('state=',state)
        #x_t = state
        u_t = np.array([u_old[0]+u[2*Nc-2,0],u_old[1]+u[2*Nc-1,0]])
        print('u_t=',u_t)
        # if check_waypoint(state,midxn,midyn):
        #     midxn,midyn = midx,midy
        if check_goal(state, goal):
            break

def main():
    cars,wheels,distance = pybullet_dynamics.sim()
    initial_state = np.array([0,0,0,0,0,0])
    goal = np.array([-3,0,0,0,0,0])
    simulate(initial_state,goal,cars,wheels,distance,Yreff)

if __name__ == '__main__':
    main()