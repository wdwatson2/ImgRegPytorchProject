

import torch


class Domain:    
    def __init__(self, omega, m, device='cpu'):
        self.device = device if device is not None else torch.device("cpu")
        self.omega = omega.to(self.device) if isinstance(omega, torch.Tensor) else torch.tensor(omega, device=self.device)
        self.dim = len(omega) // 2
        self.m = m if isinstance(m, torch.Tensor) else torch.tensor(m, device=self.device)
        self.h = torch.tensor([(self.omega[i+1] - self.omega[i]) / self.m[i//2] for i in range(0, len(self.omega), 2)], device=self.device)

    def __str__(self):
        return f"Domain: omega = {self.omega}, m = {self.m}"

    def xc(self, dim):
        return torch.linspace(self.omega[2*dim] + self.h[dim]/2, self.omega[2*dim+1] - self.h[dim]/2, self.m[dim], device=self.device)
    
    def xn(self, dim):
        return torch.linspace(self.omega[2*dim], self.omega[2*dim+1], self.m[dim]+1, device=self.device)
    
    def getCellCenteredGrid(self):
        xc =  torch.meshgrid([self.xc(i) for i in range(self.dim)],indexing='ij')
        return torch.stack(xc, dim=-1)
    
    def getNodalGrid(self):
        xn = torch.meshgrid([self.xn(i) for i in range(self.dim)],indexing='ij')
        return torch.stack(xn, dim=-1)
    
    def serialize(self):
        omega_list = self.omega.numpy() if isinstance(self.omega, torch.Tensor) else self.omega
        m_list = self.m.numpy() if isinstance(self.m, torch.Tensor) else self.m
        return {'omega': omega_list, 'm': m_list}

    def save(self, path):
        omega_list = self.omega.numpy() if isinstance(self.omega, torch.Tensor) else self.omega
        m_list = self.m.numpy() if isinstance(self.m, torch.Tensor) else self.m
        with open(path, 'w') as file:
            file.write(f"omega: {','.join(map(str, omega_list))}\n")
            file.write(f"m: {','.join(map(str, m_list))}\n")

    @classmethod
    def load(cls, path):
        with open(path, 'r') as file:
            lines = file.readlines()
        omega = list(map(float, lines[0].strip().split(': ')[1].split(',')))
        m = list(map(int, lines[1].strip().split(': ')[1].split(',')))
        return cls(torch.tensor(omega), torch.tensor(m))
    
if __name__ == '__main__':
    # Example
    omega = torch.tensor([0.0, 1.0, 0.0, 1.0,0.0, 1.0])
    m = torch.tensor([2, 3,4])
    domain = Domain(omega, m)
    print(domain)
    print(domain.xc(0))
    print(domain.xn(0))
    print(domain.getCellCenteredGrid())
    print(domain.getNodalGrid())
