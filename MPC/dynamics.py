import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random
from flax import struct
from typing import Sequence

class dynamics:
    state: Sequence[float]
    input: Sequence[float]
    inputr: Sequence[float]
    stater: Sequence[float]
    cl: int
    sf: int
    cc: int
    sr: int
    m: int
    alphaf: int
    lf: int
    lr: int
    iz: int
    
    def f1(self, x, u):
        xdot = jnp.sum(u[0])
        return xdot

    def f2(self, x, u):
        ydot = jnp.sum(x[3]*(self.lf+u[1])-self.lr*x[5])
        return ydot

    def f3(self, x, u):
        psidot = jnp.sum(x[4]/self.lr)
        return psidot

    def f4(self, x, u):
        xddot = jnp.sum((self.cl*self.sf-self.cc*self.alphaf*u[1]+self.cl*self.sr+self.m*x[4]*x[5])/self.m)
        return xddot

    def f5(self, x, u):
        yddot = jnp.sum((self.cc*self.alphaf+self.cl*self.sf*u[1]+self.cc*self.alphaf-self.m*x[3]*x[5])/self.m)
        return yddot

    def f6(self, x, u):
        psiddot = jnp.sum((self.lf*(self.cl*self.sf*u[0]+self.cc*self.alphaf)-self.cc*self.alphaf*self.lr)/self.iz)
        return psiddot
    
    def A(self, x, u):
        xr = self.stater
        ur = self.inputr
        a1 = grad(self.f1, 0)(xr, ur)
        a2 = grad(self.f2, 0)(xr, ur)
        a3 = grad(self.f3, 0)(xr, ur)
        a4 = grad(self.f4, 0)(xr, ur)
        a5 = grad(self.f5, 0)(xr, ur)
        a6 = grad(self.f6, 0)(xr, ur)

        A = jnp.concatenate((a1,a2,a3,a4,a5,a6), axis=1)
        return A
    
    def B(self, x, u):
        xr = self.xr
        ur = self.ur
        b1 = grad(self.f1, 1)(xr, ur)
        b2 = grad(self.f2, 1)(xr, ur)
        b3 = grad(self.f3, 1)(xr, ur)
        b4 = grad(self.f4, 1)(xr, ur)
        b5 = grad(self.f5, 1)(xr, ur)
        b6 = grad(self.f6, 1)(xr, ur)

        B = jnp.concatenate((b1,b2,b3,b4,b5,b6), axis=1)
        return B

    def eqn(self, x, u):
        xr = self.stater
        ur = self.inputr
        return self.A(xr, ur)*x + self.B(xr, ur)*u

    def C(self, x, u):
        C = jnp.array([[1.0, 0.0, 0.0, 0.0, 0.0],[0.0,1.0,0.0,0.0,0.0],[0.0,0.0,1.0,0.0,0.0,0.0]])
    # C = jnp.array([[1.0, 0.0, 0.0, 0.0, 0.0],[0.0,1.0,0.0,0.0,0.0],[0.0,0.0,1.0,0.0,0.0,0.0]])
    # eeta = C*state

