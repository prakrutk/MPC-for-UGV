import jax.numpy as jnp
import numpy as np
from dynamics.cardynamics import dynamics
import cvxpy as cvx
from Pybullet.racecar_differential import pybullet_dynamics
import matplotlib as plt

Nc = 5
Np = 10
initial_state = jnp.array([0,0,0,0,0,0,0,0])
x_i = jnp.array([3.0,0.0,0.0,0.0,0.0,0.0])
u_i = jnp.array([0.0,0.0])
xr = jnp.array([0.0,0.0,0.0,0.0,0.0,0.0])
ur = jnp.array([0.0,0.0])
delu = 0.1*jnp.ones((2*Nc,1))
Yreff = np.ones((3*Np,1))
Q = 100*jnp.identity(3*Np)
R = 10*jnp.identity(2*Nc)
tolerance = 0.01*jnp.ones((3*Np,1))
Ymax = Yreff + tolerance
Ymin = Yreff - tolerance
rho = 1
epi = 0
MAX_TIME = 100 # seconds

coeff = dynamics(state = x_i
                ,input = u_i
                ,inputr = ur
                ,stater = xr
                ,delu = delu
                ,cl = 1
                ,sf = 1
                ,cc = 1
                ,sr = 1
                ,m = 1
                ,alphaf = 1
                ,lf = 1
                ,lr = 1
                ,iz = 1
                ,T = 0.1
                ,Nc = Nc
                ,Np = Np)

Ynext = coeff.Y(x_i,u_i)

# Make a circle as a refernce trajectory
def reff(Np,midx,midy):
    x = midx
    y = midy
    theta = jnp.arctan2(y,x)
    xdot = 0
    ydot = 0
    thetadot = 0.0
    return x,y,theta

def stateref(t,xr):
    x = 3*jnp.cos(2*3.14*t)
    y = 3*jnp.sin(2*3.14*t)
    theta = jnp.arctan2(y,x)
    xdot = -3*jnp.sin(2*3.14*t)
    ydot = 3*jnp.cos(2*3.14*t)
    thetadot = 0.0
    xr = xr.at[0].set(x)
    xr = xr.at[1].set(y)
    xr = xr.at[2].set(theta)
    xr = xr.at[3].set(xdot)
    xr = xr.at[4].set(ydot)
    xr = xr.at[5].set(thetadot)
    return xr

def linearmpc(x_i,u_i,t,xr,midx,midy):
    xr = stateref(t,xr)
    u = cvx.Variable((2*Nc +1,1))
    # u_t=u_t.reshape(2,1)
    cost = 0.0
    constraints = []
    for i in range(Np):
        Yreff[3*i,0],Yreff[3*i+1,0],Yreff[3*i+2,0] = jnp.array(reff(Np,midx,midy))
    the_c=jnp.concatenate((coeff.theta(xr,ur),jnp.zeros((3*(Np-Nc),2*Nc))),axis=0)
    H = jnp.transpose(the_c).dot(Q).dot(the_c) + R 
    H = jnp.append(H,jnp.zeros((1,H.shape[1])),axis=0)
    c = jnp.zeros((H.shape[0],1))
    c = c.at[-1].set(rho)
    H = jnp.append(H,c,axis=1)
    E = coeff.phi(xr,ur).dot(jnp.concatenate((x_i-xr,u_i-ur),axis=0)).reshape(3*Np,1) - Yreff
    # print('E=',E)
    cost += cvx.quad_form(u,H) + 2*jnp.transpose(E).dot(Q).dot(the_c)@u[0:2*Nc,:]
    for k in range(Nc):
        constraints += [u[2*k,:] <= 10.5]
        constraints += [u[2*k,:] >= -10.5]
        constraints += [u[2*k+1,:] <= 0.1]
        constraints += [u[2*k+1,:] >= -0.1]
        constraints += [u_i[0] + u[2*k,:] <= 100]
        constraints += [u_i[1] + u[2*k+1,:] <= 1.5]
        constraints += [u_i[1] + u[2*k+1,:] >= -1.5]
        constraints += [u_i[0] + u[2*k,:] >= 10.5]
    constraints += [Ymin - u[2*Nc,:] <= coeff.Y(xr,ur)]
    constraints += [coeff.Y(xr,ur) <= Ymax + u[2*Nc,:]]
    
    prob = cvx.Problem(cvx.Minimize(cost), constraints)
    prob.solve()
    return u.value,u_i

def check_goal(state, goal):
    if abs(state[0] - goal[0]) < 0.05 and abs(state[1] - goal[1]) < 0.05:
        return True
    else:
        return False

def simulate(initial_state,goal,xr,cars,wheels,distance):
    goal = goal 
    state = initial_state
    time = 0.0
    j = 0
    u_t = jnp.array([0.0,0.0])
    while MAX_TIME >= time:
        u, u_old = linearmpc(state,u_t,time,xr)
        for i in range(Nc):
            # x,phi = pyconnect(2*u[2*i,0],u[2*i+1,0],wheels,car,useRealTimeSim)
            x,phi,midx,midy = pybullet_dynamics.loop(u_old[0]+u[2*i,0],200000,u_old[1]+u[2*i+1,0],wheels,cars,distance,j)
            time += 0.01
            j += 0.1
        state = jnp.array([x[0],x[1],phi[2],0.0,0.0,0.0])
        x_t = state
        u_t = jnp.array([u[2*Nc-2,0],u[2*Nc-1,0]])

        i +=1

        if check_goal(state, goal):
            goal = jnp.array([midx,midy,0.0,0.0,0.0,0.0])

def main():
    cars,wheels,distance = pybullet_dynamics.sim()
    initial_state = jnp.array([0,0,0,0,0,0])
    goal = jnp.array([0.32,15,0,0,0,0])
    simulate(initial_state,goal,xr,cars,wheels,distance)

if __name__ == '__main__':
    main()