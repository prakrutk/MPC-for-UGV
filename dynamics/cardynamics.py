# import jax.numpy as jnp
import numpy as np
from jax import grad, jit
# from jax import random
from flax import struct
from typing import Sequence

# @jit
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
        xdot = np.sum(x[3])
        return xdot

    def f2(self, x, u):
        alphaf = self.alphaf
        if x[3] == 0:
            alphaf = 0
        else:
            alphaf = (self.lr + self.lf)*x[5]/x[3] - u[1]
        ydot = np.sum(x[3]*(alphaf+u[1])-self.lr*x[5])
        # ydot = np.sum(u[0]*(alphaf+u[1])-self.lr*x[5])
        # ydot = np.sum(u[0]*(alphaf + u[1])/2)
        return ydot

    def f3(self, x, u):
        alphaf = self.alphaf
        if x[3] == 0:
            alphaf = 0
        else:
            alphaf = (self.lr + self.lf)*x[5]/x[3] - u[1]
        # psidot = np.sum(x[5])
        psidot = np.sum(x[4]/self.lr)
        # psidot = np.sum(x[3]*(alphaf + u[1])/2*self.lr)
        return psidot

    def f4(self, x, u):
        alphaf = self.alphaf
        if x[3] == 0:
            alphaf = 0
            xddot = np.sum((self.m*x[4]*x[5])/self.m + u[0]*self.T)
        else:
            # alphaf = (self.lr + self.lf)*x[5]/x[3] - u[1]
            xddot = np.sum((self.m*x[4]*x[5]-self.cc*((self.lr + self.lf)*x[5]/x[3] - u[1])*u[1])/self.m + u[0]*self.T)
        # xddot = np.sum((self.cl*self.sf-self.cc*alphaf*u[1]+self.cl*self.sr+self.m*x[4]*x[5])/self.m)

        return xddot

    def f5(self, x, u):
        alphaf = self.alphaf
        if x[3] == 0:
            # alphaf = 0
            yddot = np.sum((-self.m*x[3]*x[5])/self.m)
        else:
            # alphaf = (self.lr + self.lf)*x[5]/x[3] - u[1]
            yddot = np.sum((-self.m*x[3]*x[5]+self.cc*((self.lr + self.lf)*x[5]/x[3] - u[1]))/self.m)
        # yddot = np.sum((self.cc*alphaf+self.cl*self.sf*u[1]+self.cc*alphaf-self.m*x[3]*x[5])/self.m)
        # yddot = np.sum((-self.m*x[3]*x[5]+self.cc*alphaf)/self.m)
        return yddot

    def f6(self, x, u):
        alphaf = self.alphaf
        if x[3] == 0:
            alphaf = 0
            psiddot = np.sum((self.lf*(self.cl*self.sf*u[1]+self.cc*alphaf)-self.cc*alphaf*self.lr)/self.iz)
        else:
            # alphaf = (self.lr + self.lf)*x[5]/x[3] - u[1]
            psiddot = np.sum((self.cc*((self.lr + self.lf)*x[5]/x[3] - u[1])*(self.lr-self.lf))/self.iz) 
            # psiddot = np.sum((self.lf*(self.cc*((self.lr + self.lf)*x[5]/x[3] - u[1]))-self.cc*((self.lr + self.lf)*x[5]/x[3] - u[1])*self.lr)/self.iz)
        # psiddot = np.sum((self.lf*(self.cl*self.sf*u[1]+self.cc*alphaf)-self.cc*alphaf*self.lr)/self.iz)
        return psiddot
    
    def A(self, x, u):
        xr = x
        ur = u
        a1 = grad(self.f1, 0)(xr, ur)
        a2 = grad(self.f2, 0)(xr, ur)
        a3 = grad(self.f3, 0)(xr, ur)
        a4 = grad(self.f4, 0)(xr, ur)
        a5 = grad(self.f5, 0)(xr, ur)
        a6 = grad(self.f6, 0)(xr, ur)
        A = np.identity(6) + self.T*np.stack((a1,a2,a3,a4,a5,a6), axis=0)
        # print('A=',A)
        # print(A)
        return A
    
    def B(self, x, u):
        xr = x
        ur = u
        b1 = grad(self.f1, 1)(xr, ur)
        b2 = grad(self.f2, 1)(xr, ur)
        b3 = grad(self.f3, 1)(xr, ur)
        b4 = grad(self.f4, 1)(xr, ur)
        b5 = grad(self.f5, 1)(xr, ur)
        b6 = grad(self.f6, 1)(xr, ur)

        B = self.T*np.stack((b1,b2,b3,b4,b5,b6), axis=0)
        # print(B)
        # print('B=',B)
        return B
    
    def pri(self,x,u):
        xr = x
        ur = u
        A_p1 = np.column_stack((self.A(xr, ur), self.B(xr, ur)))
        A_p2 = np.column_stack((np.zeros((2,6)), np.identity(2)))
        # print('A_p1=',A_p1)
        # print('A_p2=',A_p2.shape)
        A_p = np.row_stack((A_p1, A_p2))

        B_p = np.row_stack((self.B(xr, ur), np.identity(2)))
        # print('A_p=',A_p)
        # print('B_p=',B_p)
        return A_p,B_p

    def eqn(self, x, u):
        xr = self.stater
        ur = self.inputr
        ur = ur.reshape((2,1))
        # print(self.A(xr, ur).dot(x-xr))
        print(self.B(xr, ur).dot(u - ur))
        # print(self.A(xr, ur).dot(x-xr) + self.B(xr, ur).dot(u - ur))
        return self.A(xr, ur).dot(x -xr) + self.B(xr, ur).dot(u - ur)

    def C(self, x, u):
        C = np.array([[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]])
        C = np.concatenate((C,np.zeros((3,5))),axis=1)
        # print('C=',C)
        return C
    
    def phi(self,x,u):
        phi = np.zeros((3,8))
        A_p, B_p = self.pri(x,u)
        C_p = self.C(x, u)
        phi = C_p.dot(A_p)
        # print('phi=',phi)
        # print(C_p.dot(np.power(A_p,1)))
        for i in range(self.Np-1):
            phi = np.append(phi,C_p.dot(np.power(A_p,(i+2))),axis=0)
            # print('i=',i,'phi=',phi)
        # print('phi=',phi.shape)
        return phi
    
    def theta(self,x,u):
        the = np.zeros((3,self.Nc*2))
        A_p, B_p = self.pri(x,u)
        C_p = self.C(x, u)
        # print(np.power(A_p,0))
        # print('C_p*B_p=',C_p.dot(np.power(A_p,0)).dot(B_p))
        # print('C_p*A_p*B_p=',C_p.dot(A_p).dot(B_p))
        row = np.zeros((3,self.Nc*2))
        for i in range(self.Nc):
            if i ==0:
                for j in range(self.Nc):
                    if j<i:
                        for k in range(i-j-1):
                            A_p += np.power(A_p,k+1)
                            # print('i=',i,'j=',j,'A_p=',A_p)
                    # print(the[3*i:3*i+3,2*j:2*j+2].shape)
                        the[:,2*j:2*j+2] = C_p.dot(A_p).dot(B_p)
                        # print('i=',i,'j=',j,'row=',row)
                    elif j==i:
                        the[:,2*j:2*j+2] = C_p.dot(B_p)
                    else:
                        the[:,2*j:2*j+2] = np.zeros((3,2))
                        # print(the)

            for j in range(self.Nc):
                if j<i:
                    for k in range(i-j-1):
                        A_p += np.power(A_p,k+1)
                        # print('i=',i,'j=',j,'k=',k,'A_p=',A_p)
                    # print(the[3*i:3*i+3,2*j:2*j+2].shape)
                    row[:,2*j:2*j+2] = C_p.dot(A_p).dot(B_p)
                    # print('i=',i,'j=',j,'row=',row)
                elif j==i:
                    row[:,2*j:2*j+2] = C_p.dot(B_p)
                else:
                    row[:,2*j:2*j+2] = np.zeros((3,2))
                    # print('i=',i,'row=',row)
                # print(i)
                # print('i=',i,'the=',the)
            else: 
                # print('i=',i,'the=',the)
                the = np.append(the,row,axis=0)
                # print(i)
        # the = np.flip(the,axis=1)

        # print('the=',the)
        return the
    
    def Y(self,x,u,delu):
        stated = self.state - x
        inputd = self.input - u
        # print('stated=',stated)
        # print('inputd=',inputd)
        # print('delu=',delu)
        # print('np.concatenate((stated,inputd),axis=0)=',np.concatenate((stated,inputd),axis=0))
        Y1 = self.phi(x,u).dot(np.concatenate((x,u),axis=0)).reshape((3*self.Np,1)) 
        Y2 = self.theta(x,u).dot(delu)
        print(self.theta(x,u))
        Y2 = np.append(Y2,np.zeros(((Y1.shape[0] - Y2.shape[0]),1)),axis=0)
        Y = Y1 + Y2
        return Y,Y1,Y2
