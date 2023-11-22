import jax.numpy as jnp
from dynamics.cardynamics import dynamics
import cvxpy as cvx
from Pybullet.racecar_differential import pybullet_dynamics

Nc = 5
Np = 10
initial_state = jnp.array([0,0,0,0,0,0,0,0])
x_t = jnp.array([0.0,0.0,0.0,0.0,0.0,0.0])
u_t = jnp.array([0.0,0.0])
xr = jnp.array([1.5,0.5,0.0,0.0,0.0,0.0])
ur = jnp.array([0.0,0.0])
delu = 0.1*jnp.ones((2*Nc,1))
Yreff = jnp.ones((3*Np,1))
Q = 100*jnp.identity(3*Np)
R = 10*jnp.identity(2*Nc)
tolerance = 0.01*jnp.ones((3*Np,1))
Ymax = Yreff + tolerance
Ymin = Yreff - tolerance
rho = 1
epi = 0
MAX_TIME = 100 # seconds

class state:

    def __init__(self, x, y, theta, xdot, ydot, thetadot):
        self.x = x
        self.y = y
        self.theta = theta
        self.xdot = xdot
        self.ydot = ydot
        self.thetadot = thetadot

coeff = dynamics(state = x_t
                ,input = u_t
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

Ynext = coeff.Y(x_t,u_t)


def linearmpc(x,u_t):
    u = cvx.Variable((2*Nc +1,1))
    # u_t=u_t.reshape(2,1)
    cost = 0.0
    constraints = []
    the_c=jnp.concatenate((coeff.theta(),jnp.zeros((3*(Np-Nc),2*Nc))),axis=0)
    H = jnp.transpose(the_c).dot(Q).dot(the_c) + R 
    H = jnp.append(H,jnp.zeros((1,H.shape[1])),axis=0)
    c = jnp.zeros((H.shape[0],1))
    c = c.at[-1].set(rho)
    H = jnp.append(H,c,axis=1)
    E = coeff.phi().dot(jnp.concatenate((x_t-xr,u_t-ur),axis=0)).reshape(3*Np,1) - Yreff
    # print('E=',E)
    cost += cvx.quad_form(u,H) + jnp.transpose(E
                                               
                                               
                                               ).dot(Q).dot(the_c)@u[0:2*Nc,:]
    for k in range(2*Nc):
        constraints += [u[k,:] <= 0.5]
        constraints += [u[k,:] >= -0.5]
        #constraints += [u_t[k_t,:] <= 5]
    constraints += [Ymin - u[2*Nc,:] <= coeff.Y(x,u)]
    constraints += [coeff.Y(x,u) <= Ymax + u[2*Nc,:]]
    
    prob = cvx.Problem(cvx.Minimize(cost), constraints)
    prob.solve()
    return u.value

def check_goal(state, goal):
    if abs(state[0] - goal.x) < 0.05 and abs(state[1] - goal.y) < 0.05:
        return True
    else:
        return False

def traj_gen(start,goal):
    x = start.x
    y = start.y
    theta = start.theta
    xdot = start.xdot
    ydot = start.ydot
    thetadot = start.thetadot
    xg = goal.x
    yg = goal.y
    thetag = goal.theta
    xdotg = goal.xdot
    ydotg = goal.ydot
    thetadotg = goal.thetadot
    t = 0
    while t < MAX_TIME:
        x = x + xdot*0.1
        y = y + ydot*0.1
        theta = theta + thetadot*0.1
        xdot = xdot + xdotg*0.1
        ydot = ydot + ydotg*0.1
        thetadot = thetadot + thetadotg*0.1
        t += 0.1
        yield state(x,y,theta,xdot,ydot,thetadot)

def simulate(initial_state,goal):
    goal = goal 
    state = initial_state
    time = 0.0

    cars,wheels,distance = pybullet_dynamics.sim()

    u = u_t
    while MAX_TIME >= time:
        u = linearmpc(state,u)
        for i in range(Nc):
            # x,phi = pyconnect(2*u[2*i,0],u[2*i+1,0],wheels,car,useRealTimeSim)
            x,phi = pybullet_dynamics.loop(2*u[2*i,0],20,u[2*i+1,0],wheels,cars,distance)
        state = jnp.array([x[0],x[1],phi[-1],0,0,0])
        x_t = state
        time += 0.1
        i +=1

        if check_goal(state, goal):
            print("Goal")
            break

def main():
    initial_state = state(0,0,0,0,0,0)
    goal = state(1.5,0.5,0,0,0,0)
    simulate(initial_state,goal)

if __name__ == '__main__':
    main()