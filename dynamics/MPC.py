import jax.numpy as jnp
from dynamics.cardynamics import dynamics
import cvxpy as cvx

Nc = 5
Np = 10
initial_state = jnp.array([0,0,0,0,0,0,0,0])
x = jnp.array([0.0,0.0,0.0,0.0,0.0,0.0])
u = jnp.array([0.0,0.0])
xr = jnp.array([0.0,0.0,0.0,0.0,0.0,0.0])
ur = jnp.array([0.0,0.0])
delu = 0.1*jnp.ones((2*Nc,1))
Yreff = jnp.ones((3*Np,1))
Q = 100*jnp.identity(6)
R = 10*jnp.identity(2)
rho = 1
epi = 0
coeff = dynamics(state = x
                ,input = u
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

Ynext = coeff.Y(x,u)
E = coeff.phi().dot(jnp.concatenate((x-xr,u-ur),axis=0))- Yreff
print('successfully executed')

# def linearmpc(x,u):
#     u = cvx.Variable((2*Nc +1,1))

#     cost = 0.0
#     constraints = []

#     for k in range(Nc):
#         cost += cvx.quad_form(E[:,k],Q)
#         cost += cvx.quad_form(u[:,k],R)
#         constraints += [Ynext[:,k] == coeff.Y(x,u)]
#         constraints += [u[:,k] <= 0.5]
#         constraints += [u[:,k] >= -0.5]
    
#     prob = cvx.Problem(cvx.Minimize(cost), constraints)
#     prob.solve()
#     return u.value

# print(linearmpc(x,u))
