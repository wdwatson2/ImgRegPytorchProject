import torch

# this class defines the L2 distance measure between two images given as pytorch tensord
class SSDDistance(torch.nn.Module):
    def __init__(self,domain):
        super().__init__()
        self.domain = domain
    
    def forward(self, T, R):
        res = T.view(-1)-R.view(-1)        
        return torch.prod(self.domain.h)*torch.norm(res)**2
    
    def __str__(self):
        return "L2Distance"

# The following is rough but allows regularization based on the norm of all velocities.
# A better approach is to only sum the norm of the velocities along the trajectory (should be much less expensive as well).
class RegularizedDistance(torch.nn.Module):
    def __init__(self, domain, velocity_model, lambdas=0):
        super().__init__()
        self.domain = domain
        self.velocity_model = velocity_model
        self.l2 = lambdas

    def forward(self, T, R, x, times):
        res = T.view(-1) - R.view(-1)
        ssd_distance = torch.prod(self.domain.h) * torch.norm(res)**2

        # approximate the integral with trapezoidal rule
        reg_l2 = 0.0

        for i,t in enumerate(times):
            velocity = self.velocity_model(t, x) # (num_points, 2)
            l2_reg = torch.norm(velocity)**2
            
            # endpoints are not doubled in trapezoidal rule
            if i == 0 or i == len(times)-1:
                reg_l2 += l2_reg
            else:
                reg_l2 += 2*l2_reg

        h = times[1] - times[0]
        reg_l2 *= h/2 

        regularized_ssd = ssd_distance + self.l2*reg_l2

        return regularized_ssd


class trajectoryDistance(torch.nn.Module):
    def __init__(self, domain, lambdas=0):
        super().__init__()
        self.domain = domain
        self.l2 = lambdas

    def forward(self, T, R, length):
        res = T.view(-1) - R.view(-1)
        ssd_distance = torch.prod(self.domain.h) * torch.norm(res)**2

        regularized_ssd = ssd_distance + self.l2*torch.norm(length)

        return regularized_ssd