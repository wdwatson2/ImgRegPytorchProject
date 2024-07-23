import torch
import torch.nn as nn
from torch.nn.functional import pad
from src.domain import Domain   
import matplotlib.pyplot as plt
from src.plotting import view_image_2d, view_image_3d   



class SplineInter(nn.Module):
    def __init__(self, image, domain, regularizer=None, theta=0.0):
        super().__init__()
        self.image = image
        self.domain = domain
        self.pad = 2
        self.regularizer = regularizer
        self.theta = theta
        self.coeffs = self.get_coefficients(image)
        self.coeffs = pad(self.coeffs, (self.pad, self.pad) * self.domain.dim, mode='constant', value=0)

    def __str__(self):
        return "SplineInter: image = " + str(self.image.shape) + ", domain = " + str(self.domain)        

    def b0(self,j,xi):
        if j == 1:
            return (2 + xi) *(2 + xi) *(2 + xi)
        elif j == 2:
            return -(3 * xi + 6) * (xi * xi) + 4
        elif j == 3:
            return (3 * xi - 6) * (xi * xi) + 4
        elif j == 4:
            return ((2 - xi) *(2 - xi) *(2 - xi))

    def forward(self,x):
        x = x.view(-1,self.domain.dim)
        # map x from [omega[0]+h/2, omega[1-h/2] -> [0,m-1]
        x = (x - self.domain.omega[0:-1:2].view(1,-1))/self.domain.h.view(1,-1)-0.5

        def Valid(i):
            return (-2<x[:,i]) & (x[:,i]<self.domain.m[i])

        valid = Valid(0)
        for i in range(1, self.domain.dim):
            valid = valid & Valid(i)
        
        P = torch.floor(x).to(torch.long)
        x = x - P

        out = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
        coeffs = self.coeffs.view(-1)
        if self.domain.dim==1:
            p = self.pad + P[valid]
            xi = x[valid]            
            out[valid] = coeffs[p+2] * self.b0(1,xi-2) + coeffs[p+1] * self.b0(2,xi-1) + coeffs[p] * self.b0(3,xi) + coeffs[p-1] * self.b0(4,xi+1)
            
        elif self.domain.dim==2:
            i1 = self.domain.m[1] + 2 * self.pad
            i2 = 1

            p =  i1*(self.pad + P[valid,0]) + i2*(self.pad+P[valid,1])
            x1 = x[valid,0]
            x2 = x[valid,1]
            
            for j1 in range(-1,3):
                for j2 in range(-1,3):
                    out[valid] = out[valid] + (coeffs[p+j1*i1+j2*i2] * self.b0(3-j1,x1-j1) * self.b0(3-j2,x2-j2)).view(-1,1)
            
            
        elif self.domain.dim==3:
            i3 = 1
            i2 = self.domain.m[2]+2*self.pad
            i1 = i2 * (self.domain.m[1] + 2 * self.pad)

            p = i1*(self.pad + P[valid, 0]) + i2 * (self.pad + P[valid, 1]) + i3 * (self.pad + P[valid, 2])
            x1 = x[valid, 0]
            x2 = x[valid, 1]
            x3 = x[valid, 2]
            b1 = []
            b2 = []
            b3 = []
            for j1 in range(-1,3):
                b1.append(self.b0(3 - j1, x1 - j1))
            for j2 in range(-1, 3):
                b2.append(self.b0(3 - j2, x2 - j2))
            
            for j3 in range(-1, 3):
                b3.append(self.b0(3 - j3, x3 - j3))            

            for j1 in range(-1,3):
                for j2 in range(-1,3):
                    for j3 in range(-1,3):
                        Phij = coeffs[p + j1 * i1 + j2 * i2 + j3 * i3]
                        out[valid] = out[valid] + (Phij * b1[j1+1] * b2[j2+1] * b3[j3+1]).view(-1,1)
        
        return out

    def get_coefficients(self, T):
        device = T.device  # Ensure we're using the same device as T
        dtype = T.dtype

        def getB(d):
            # Ensure matrices are created on the same device and dtype as T
            B = torch.diag(torch.ones(self.domain.m[d], device=device, dtype=dtype)) * 4 + \
                torch.diag(torch.ones(self.domain.m[d]-1, device=device, dtype=dtype), 1) + \
                torch.diag(torch.ones(self.domain.m[d]-1, device=device, dtype=dtype), -1)
            
            if self.regularizer is not None and self.theta > 0:
                if self.regularizer == "moments":
                    P = torch.diag(torch.ones(self.domain.m[d], device=device, dtype=dtype)) * 96 - \
                        torch.diag(torch.ones(self.domain.m[d]-1, device=device, dtype=dtype), 1) * 54 - \
                        torch.diag(torch.ones(self.domain.m[d]-1, device=device, dtype=dtype), -1) * 54 + \
                        torch.diag(torch.ones(self.domain.m[d]-3, device=device, dtype=dtype), 3) * 6 + \
                        torch.diag(torch.ones(self.domain.m[d]-3, device=device, dtype=dtype), -3) * 6
                    M = B.t() @ B + self.theta * P
            else:
                M = B.t() @ B

            return B, M

        # Ensuring the reshape and permutation operations are also considering the device and dtype
        if self.domain.dim == 1:
            B, M = getB(0)
            W = torch.linalg.solve(M, B.t() @ T)
        elif self.domain.dim == 2:
            B, M = getB(0)
            W = torch.linalg.solve(M, B.t() @ T)
            B, M = getB(1)
            W = torch.linalg.solve(M, B.t() @ W.permute(1, 0)).permute(1, 0)
        elif self.domain.dim == 3:
            B, M = getB(0)
            W = torch.linalg.solve(M, B.t() @ T.reshape(self.domain.m[0], -1)).reshape(tuple(self.domain.m))
            B, M = getB(1)
            W = torch.linalg.solve(M, B.t() @ W.permute(1, 0, 2).reshape(self.domain.m[1], -1)).reshape(self.domain.m[1], self.domain.m[0], self.domain.m[2]).permute(1, 0, 2)
            B, M = getB(2)
            W = torch.linalg.solve(M, B.t() @ W.permute(2, 0, 1).reshape(self.domain.m[2], -1)).reshape(self.domain.m[2], self.domain.m[0], self.domain.m[1]).permute(1, 2, 0)

        return W

# inherit SplineInter into a class SplintInterVmap. The only thing we'll change is the forward. Just copy the old one for now
class SplineInterVmap(SplineInter):

    # consstructor
    def __init__(self, image, domain, regularizer=None, theta=0.0):
        super().__init__(image, domain, regularizer, theta)

    # forward function
    def forward(self, x):
        x = x.view(-1,self.domain.dim)

        # map x from [omega[0]+h/2, omega[1-h/2] -> [0,m-1]
        x = (x - self.domain.omega[0:-1:2].view(1,-1))/self.domain.h.view(1,-1)-0.5
        valid =  torch.all((-2<x) & (x<self.domain.m.view(1,-1)),dim=1)
        # clip v to be between 0 and m-1
        for i in range(self.domain.dim):
            x[:,i] = torch.clamp(x[:,i],0,self.domain.m[i]-1)
        P = torch.floor(x).to(torch.long)
        x = x - P

        out = torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)
        coeffs = self.coeffs.view(-1)
        if self.domain.dim==1:
            p = self.pad + P
            xi = x
            out = coeffs[p+2] * self.b0(1,xi-2) + coeffs[p+1] * self.b0(2,xi-1) + coeffs[p] * self.b0(3,xi) + coeffs[p-1] * self.b0(4,xi+1)
        
        elif self.domain.dim==2:
            i1 = self.domain.m[1] + 2 * self.pad
            i2 = 1

            p =  i1*(self.pad + P[:,0]) + i2*(self.pad+P[:,1])
            x1 = x[:,0]
            x2 = x[:,1]
            
            for j1 in range(-1,3):
                for j2 in range(-1,3):
                    out = out + (coeffs[p+j1*i1+j2*i2] * self.b0(3-j1,x1-j1) * self.b0(3-j2,x2-j2)).view(-1,1)
            
            
        elif self.domain.dim==3:
            # throw an error if the dimension is not 3
            raise ValueError("Dimension 3 nyi")
            
        return valid * out





if  __name__ =='__main__':
        domain_data = Domain(torch.tensor([0.0, 2.0]), torch.tensor([5]))
        xc = domain_data.getCellCenteredGrid()
        Td = torch.tensor([1.0, 3.0, -1.0, 0.5, 0.0])

        inter = SplineInter(Td, domain_data, regularizer="moments", theta=1e-6)
        domain_fine = Domain(torch.tensor([0.5,1.5]), torch.tensor([100]))
        xc_fine = domain_fine.getCellCenteredGrid()
        Td_fine = inter(xc_fine)

        plt.Figure()
        plt.subplot(2,3,1)
        plt.plot(xc,Td,'bs')
        plt.plot(xc_fine,Td_fine,'-r')
        
        domain_data = Domain(torch.tensor([0.0, 2.0, 0, 1.0]), torch.tensor([3, 2]))
        Td = torch.tensor([[-20.0,2.0],[3.0,4.0],[5.0,60.0]])
        inter = SplineInter(Td, domain_data)
        domain_fine = Domain(torch.tensor([0.0, 2.0, 0, 1.0]), torch.tensor([100, 100]))
        xc_fine = domain_fine.getCellCenteredGrid()
        Td_fine = inter(xc_fine)
        plt.subplot(2,3,2)
        view_image_2d(Td, domain_data)
        plt.colorbar()
        plt.subplot(2,3,3)
        view_image_2d(Td_fine, domain_fine)
        plt.colorbar()
        
        domain_data = Domain(torch.tensor([0.0, 2.0, 0, 1.0,0.0, 0.6]), torch.tensor([8,16,32]))
        Td = torch.randn(tuple(domain_data.m))
        domain_fine = Domain(torch.tensor([0.0, 2.0, 0, 1.0,0.0, 0.6]), torch.tensor([100, 100, 100]))
        xc_fine = domain_fine.getCellCenteredGrid()
        inter = SplineInter(Td, domain_data, regularizer="moments", theta=1e-6)
        Td_fine = inter(xc_fine)
        plt.subplot(2,3,4, projection='3d')
        view_image_3d(Td, domain_data)
        plt.colorbar()
        plt.subplot(2,3,5, projection='3d')
        view_image_3d(Td_fine, domain_fine)
        plt.colorbar()

        

        plt.show()
        
        

