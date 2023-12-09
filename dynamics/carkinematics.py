import numpy as np
from flax import struct
from typing import Sequence
import math

class Kinematics(struct.PyTreeNode):
    state: Sequence[float]
    input: Sequence[float]
    inputr: Sequence[float]
    stater: Sequence[float]
    delu: Sequence[float]
    lf: float
    lr: float
    T: float
    Nc: int
    Np: int

    def A(self, x, u):
        A = np.zeros((4, 4))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.T * math.cos(x[3])
        A[0, 3] = - self.T * x[2] * math.sin(x[3])
        A[1, 2] = self.T * math.sin(x[3])
        A[1, 3] = self.T * x[2] * math.cos(x[3])
        A[3, 2] = self.T * math.tan(u[1]) / (self.lf + self.lr)
        return A
    
    def B(self, x, u):
        B = np.zeros((4, 2))
        B[2, 0] = self.T
        B[3, 1] = self.T * x[2] / ((self.lr+self.lf) * math.cos(u[1]) ** 2)
        # print('B=',B)
        return B
    
    def C(self, x, u):
        C = np.zeros((4,1))
        C[0] = self.T * x[2] * math.sin(x[3]) * x[3]
        C[1] = - self.T * x[2] * math.cos(x[3]) * x[3]
        C[3] = - self.T * x[2] * u[1] / ((self.lr + self.lf) * math.cos(u[1]) ** 2)
        return C
    
    def D(self, x, u):
        D = np.zeros((3,4))
        D[0, 0] = 1
        D[1, 1] = 1
        D[2, 3] = 1
        return D
    
    def pri(self, x, u):
        A_p1 = np.column_stack((self.A(x,u),self.B(x,u)))
        A_p2 = np.column_stack((np.zeros((2,4)),np.identity(2)))
        A_p = np.row_stack((A_p1,A_p2))
        B_p = np.row_stack((self.B(x,u),np.identity(2)))
        C_p = np.row_stack((self.C(x,u),np.zeros((2,1))))
        D_p = np.column_stack((self.D(x,u),np.zeros((3,2))))
        # print('A_p=',A_p)
        # print('B_p=',B_p)
        # print('C_p=',C_p)
        # print('D_p=',D_p)
        return A_p,B_p,C_p,D_p
    
    def theta(self,x,u):
        the = np.zeros((3,self.Nc*2))
        A_p, B_p, C_p, D_p = self.pri(x,u)
        row = np.zeros((3,self.Nc*2))
        for i in range(self.Nc):
            for j in range(self.Nc):
                if j<i:
                    row[:,2*j:2*j+2] = (D_p.dot(np.power(A_p,(i-j))).dot(B_p))
                    # print('i=',i,'j=',j,'row=',row)
                elif j==i:
                    row[:,2*j:2*j+2] = D_p.dot(B_p)
                    # print('i=',i,'j=',j,'row=',row)
                else:
                    row[:,2*j:2*j+2] = np.zeros((3,2))
                    # print('i=',i,'j=',j,'row=',row)
            if i == 0:
                the = row
            else:
                the = np.append(the,row,axis=0)
        return the
    
    def phi(self,x,u):
        phi = np.zeros((3,6))
        A_p, B_p, C_p, D_p = self.pri(x,u)
        phi = D_p.dot(A_p)
        for i in range(self.Np-1):
            phi = np.append(phi,D_p.dot(np.power(A_p,(i+2))),axis=0)
            # print('i=',i,'phi=',phi)
        return phi
    
    def psi(self,x,u):
        psi = np.zeros((3,1))
        A_p, B_p, C_p, D_p = self.pri(x,u)
        psi = D_p.dot(C_p)
        A = A_p
        for i in range(self.Np-1):
            for j in range(i+1):
                A += np.power(A,j) 
            psi = np.append(psi,D_p.dot(A).dot(C_p),axis=0)
        return psi
    
    def Y(self,x,u,delu):
        # A_p, B_p, C_p = self.pri(x,u)
        phi = self.phi(x,u)
        theta = self.theta(x,u)
        psi = self.psi(x,u)
        # print('phi=',phi)
        # print('theta=',theta)
        # print('psi=',psi)
        Y1 = phi.dot(np.concatenate((x,u),axis=0))
        Y2 = theta.dot(delu)
        # print(theta)
        # print(delu)
        # print('Y2=',Y2)
        Y2 = np.append(Y2,np.zeros((Y1.shape[0]-Y2.shape[0],Y2.shape[1])),axis=0)
        Y = Y1.reshape(3*self.Np,1) + Y2
        # print('Y=',Y1.shape)
        # Y1 = phi
        # Y2 = theta
        return Y,Y1,Y2,psi
    