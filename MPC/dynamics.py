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
    delu: Sequence[float]
    cl: float
    sf: float
    cc: float
    sr: float
    m: float
    alphaf: float
    lf: float
    lr: float
    iz: float
    T: float
    Nc: int
    Np: int
    
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

        A = jnp.identity(6) + self.T*jnp.concatenate((a1,a2,a3,a4,a5,a6), axis=1)
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

        B = self.T*jnp.concatenate((b1,b2,b3,b4,b5,b6), axis=1)
        return B
    
    def pri(self):
        xr = self.stater
        ur = self.inputr
        A_p1 = jnp.concatenate((self.A(xr, ur), self.B(xr, ur)), axis=1)
        A_p2 = jnp.concatenate((jnp.zeros((2,6)), jnp.identity(2)), axis=1)
        A_p = jnp.concatenate((A_p1, A_p2), axis=0)

        B_p = jnp.concatenate((self.B(xr, ur), jnp.identity(2)), axis=0)
        return A_p,B_p

    def eqn(self, x, u):
        xr = self.stater
        ur = self.inputr
        return self.A(xr, ur)*(x -xr) + self.B(xr, ur)*(u - ur)

    def C(self, x, u):
        C = jnp.array([[1.0,0.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0,0.0],[0.0,0.0,1.0,0.0,0.0,0.0]])
        return jnp.concatenate(C,jnp.zeros((3,2)),axis=1)
    
    def theta(self):
        the = jnp.array([])
        A_p, B_p = self.pri()
        for i in range(self.Np-1):
            the = jnp.concatenate(the,jnp.power(A_p,(i+1)),axis=0)
        return the
    
    def phi(self):
        phi = jnp.array([])
        A_p, B_p = self.pri()
        for i in range(self.Nc):
            row = jnp.zeros(1,self.Nc)
            for j in range(i):
                row[j] = jnp.power(A_p,(j))*B_p
            phi = jnp.concatenate(phi,row,axis=0)
        return phi
    
    def Y(self):
        Y = self.C*(self.theta*jnp.concatenate(self.state-self.stater,self.input-self.inputr,axis=0) + self.phi*self.delu)
        return Y
    
    

