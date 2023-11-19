import jax.numpy as jnp
from dynamics.cardynamics import dynamics
import cvxpy as cvx

Nc = 5
Np = 10
initial_state = jnp.array([0,0,0,0,0,0,0,0])
x_t = jnp.array([0.0,0.0,0.0,0.0,0.0,0.0])
u_t = jnp.array([0.0,0.0])
xr = jnp.array([0.0,0.0,0.0,0.0,0.0,0.0])
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
E = coeff.phi().dot(jnp.concatenate((x_t-xr,u_t-ur),axis=0)).reshape(3*Np,1) - Yreff
print('successfully executed')

def linearmpc(x,u_t):
    u = cvx.Variable((2*Nc +1,1))
    u_t=u_t.reshape(2,1)
    cost = 0.0
    constraints = []
    the_c=jnp.concatenate((coeff.theta(),jnp.zeros((3*(Np-Nc),2*Nc))),axis=0)
    H = jnp.transpose(the_c).dot(Q).dot(the_c) + R 
    H = jnp.append(H,jnp.zeros((1,H.shape[1])),axis=0)
    c = jnp.zeros((H.shape[0],1))
    c = c.at[-1].set(rho)
    H = jnp.append(H,c,axis=1)
    cost += cvx.quad_form(u,H) + jnp.transpose(E).dot(Q).dot(the_c)*u[0:2*Nc,:]
    for k in range(2*Nc):
        constraints += [u[k,:] <= 0.5]
        constraints += [u[k,:] >= -0.5]
        #constraints += [u_t[k_t,:] <= 5]
    constraints += [Ymin - u[2*Nc,:] <= coeff.Y(x,u)]
    constraints += [coeff.Y(x,u) <= Ymax + u[2*Nc,:]]
    
    prob = cvx.Problem(cvx.Minimize(cost), constraints)
    prob.solve()
    return u.value

print(linearmpc(x_t,u_t))

def simulate(initial_state,goal):
    goal = goal 
    state = initial_state
    x = [state.x]
    y = [state.y]
    theta = [state.theta]
    xdot = [state.xdot]
    ydot = [state.ydot]
    thetadot = [state.thetadot]
    u = u_t
    for i in range(100):
        u = linearmpc(x,u)
        x = coeff.evol(x,u)
        print(x)