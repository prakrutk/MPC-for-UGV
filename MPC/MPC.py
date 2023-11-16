import jax.numpy as jnp
from MPC.dynamics import dynamics, eqn

x = jnp.array([0,0,0,0,0,0])
u = jnp.array([0,0])
xr = jnp.array([0,0,0,0,0,0])
ur = jnp.array([0,0])
coeff = dynamics(state = x
                 ,input = u
                ,inputr = ur
                ,stater = xr
                ,cl = 1
                ,sf = 1
                ,cc = 1
                ,sr = 1
                ,m = 1
                ,alphaf = 1
                ,lf = 1
                ,lr = 1
                ,iz = 1)

xnext = coeff.eqn(x,u)