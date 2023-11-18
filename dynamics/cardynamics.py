import jax.numpy as jnp
import numpy as np
from jax import grad, jit, vmap
from jax import random
from flax import struct
from typing import Sequence

class dynamics(struct.PyTreeNode):
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
        A = jnp.identity(6) + self.T*np.stack((a1,a2,a3,a4,a5,a6), axis=1)
        # A = np.concatenate((a1,a2,a3,a4,a5,a6), axis=1)
        # print(A)
        return A
    
    def B(self, x, u):
        xr = self.stater
        ur = self.inputr
        b1 = grad(self.f1, 1)(xr, ur)
        b2 = grad(self.f2, 1)(xr, ur)
        b3 = grad(self.f3, 1)(xr, ur)
        b4 = grad(self.f4, 1)(xr, ur)
        b5 = grad(self.f5, 1)(xr, ur)
        b6 = grad(self.f6, 1)(xr, ur)

        B = self.T*np.stack((b1,b2,b3,b4,b5,b6), axis=0)
        return B
    
    def pri(self):
        xr = self.stater
        ur = self.inputr
        A_p1 = np.column_stack((self.A(xr, ur), self.B(xr, ur)))
        A_p2 = np.column_stack((jnp.zeros((2,6)), jnp.identity(2)))
        A_p = np.row_stack((A_p1, A_p2))

        B_p = np.row_stack((self.B(xr, ur), jnp.identity(2)))
        return A_p,B_p

    def eqn(self, x, u):
        xr = self.stater
        ur = self.inputr
        return self.A(xr, ur)*(x -xr) + self.B(xr, ur)*(u - ur)

    def C(self, x, u):
        C = jnp.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
        C = jnp.concatenate((C,jnp.zeros((5,3))),axis=0)
        return jnp.concatenate((C,jnp.zeros((8,5))),axis=1)
    
    def phi(self):
        phi = jnp.zeros((8,8))
        A_p, B_p = self.pri()
        C_p = self.C(self.stater, self.inputr)
        phi = C_p.dot(A_p)
        for i in range(self.Np-2):
            phi = np.append(phi,C_p.dot(jnp.power(A_p,(i+2))),axis=0)
        return phi
    
    def theta(self):
        the = jnp.zeros((8,2))
        A_p, B_p = self.pri()
        C_p = self.C(self.stater, self.inputr)
        row = np.zeros((8,self.Nc*2))
        for i in range(self.Nc-1):
            for j in range(self.Nc-1):
                if j<=i:
                    row[:,2*j:2*j+2] = C_p.dot(np.power(A_p,(i-j)).dot(B_p)) 
                else:
                    row[:,2*j:2*j+2] = jnp.zeros((8,2))
            if i==0:
                the = row
            else:    
                the = np.append(the,row,axis=0)
        return the
    
    def Y(self,x,u):
        stated = self.state - self.stater
        inputd = self.input - self.inputr
        print(jnp.concatenate((stated,inputd),axis=0).shape)
        Y1 = self.phi()*jnp.concatenate((stated,inputd),axis=0) 
        Y2 = self.theta().dot(self.delu)
        Y2 = np.append(Y2,jnp.zeros(((Y1.shape[0] - Y2.shape[0]),1)),axis=0)
        Y = Y1 + Y2
        return Y
