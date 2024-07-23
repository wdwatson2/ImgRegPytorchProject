import torch
import torchdiffeq
import torch.nn as nn
from src.interpolation import SplineInter
import copy

class Affine2d(nn.Module):
    def __init__(self):
        super().__init__()

        # self.precondition = precondition

        # set self.theta as a scalar parameter, initialize with 0
        self.A = nn.Parameter(torch.eye(2, 2)) 
        self.b = nn.Parameter(torch.zeros(1,2))
        
    def __str__(self):
        return "Affine2D: A = " + str(self.A ) + ", b = " + str(self.b )


    def forward(self, x):
        """
        Apply the transformation to x using the formula
        y = Ax + b
        """
        x = x.view(-1,2)
        return (torch.matmul(x, self.A.t()) + self.b)
    
    
    def inverse(self, x):
        """
        Apply the inverse transformation to x using the formula
        y = A^{-1} (x - b)
        """
        x = x.view(-1,2)
        return torch.matmul(x - self.b, torch.inverse(self.A.t())) 
    
    def invert(self):
        # Explicitly copy parameters and necessary internal states
        self.b = nn.Parameter(torch.matmul(- self.b, torch.inverse(self.A.t())).reshape((1,2)).contiguous())
        self.A = nn.Parameter(torch.inverse(self.A).reshape((2,2)).contiguous())
    
    def clone(self):
        # Create a new instance of Affine2d
        new_instance = Affine2d()
        # Explicitly copy parameters and necessary internal states
        
        new_instance.A = nn.Parameter(self.A.clone())
        new_instance.b = nn.Parameter(self.b.clone())
        
        return new_instance
    
    def __add__(self, other):
        if isinstance(other, Affine2d):
            new_A = self.A + other.A
            new_b = self.b + other.b
            new_transform = Affine2d()
            new_transform.A = nn.Parameter(new_A)
            new_transform.b = nn.Parameter(new_b)
            return new_transform
        else:
            raise ValueError("Operand must be an instance of Affine2d")

    def __mul__(self, scalar):
        if isinstance(scalar, (int, float)):
            new_A = self.A * scalar
            new_b = self.b * scalar
            new_transform = Affine2d()
            new_transform.A = nn.Parameter(new_A)
            new_transform.b = nn.Parameter(new_b)
            return new_transform
        else:
            raise ValueError("Operand must be a scalar (int or float)")
    


class Rigid2d(nn.Module):
    def __init__(self):
        super().__init__()

        # set self.theta as a scalar parameter, initialize with 0
        self.theta = nn.Parameter(torch.zeros(1))
        self.b = nn.Parameter(torch.zeros(1,2))

        
        
    def __str__(self):
        return "Rigid2d: theta = " + str(self.theta) + ", b = " + str(self.b)
    
    def Q(self):
        Q = torch.stack([torch.stack([torch.cos(self.theta), -torch.sin(self.theta)]),
                         torch.stack([torch.sin(self.theta), torch.cos(self.theta)])]).squeeze(2)
        return Q

    def forward(self, x):
        """
        Apply the transformation to x using the formula
        y = Q(theta) x + b
        """
        x = x.view(-1,2)
        return torch.matmul(x, self.Q().t()) + self.b
    
    
    def inverse(self, x):
        """
        Apply the inverse transformation to x using the formula
        y = Q(theta)^{-1} (x - b)
        """
        x = x.view(-1,2)
        return torch.matmul(x - self.b, self.Q()) 

class VelocityNN(nn.Module):

    def __init__(self, dim, widths, seed=None):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)
        self.dim = dim
        self.widths = widths
        self.layers = nn.ModuleList()
        for i in range(len(widths)-1):
            self.layers.append(nn.Linear(widths[i], widths[i+1]))
        self.layers.append(nn.Linear(widths[-1], dim))
        
    def forward(self, t, x):
        """
        Apply the neural network to the input x at time t
        """
        x = x.view(-1, self.dim)
        x = torch.cat([x,torch.full_like(x[:, :1], fill_value=t)], dim=1)
        
        for i in range(len(self.layers)-1):
            x = torch.tanh(self.layers[i](x))
        vel = self.layers[-1](x)
        return vel

    def __str__(self):
        return "VelocityNN: dim = " + str(self.dim) + ", widths = " + str(self.widths)
    
    def clone(self):
        # Create a new instance of VelocityNN with the same dimensions and layer widths
        new_instance = VelocityNN(self.dim, self.widths)
        # Manually copy the parameters from the existing layers to the new instance layers
        for original_layer, new_layer in zip(self.layers, new_instance.layers):
            new_layer.load_state_dict(original_layer.state_dict())
        return new_instance


class NeuralLengthRegularize(nn.Module):
    def __init__(self, velocity, method='rk4', t=torch.linspace(0.0, 1.0, 10)):
        super().__init__()
        self.velocity = velocity
        self.method = method
        self.t = t

    def forward(self, x):
        # Assume x initially does not have the accumulated length
        # Augment x with an initial length of 0
        initial_length = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
        x_augmented = torch.cat([x, initial_length], dim=1)

        def vel_len(t, x_aug):
            # Split the augmented state into position and accumulated length
            x, length = x_aug[:, :-1], x_aug[:, -1:]

            vel = self.velocity(t, x) 

            vel_norm = torch.linalg.norm(vel, dim=1, keepdim=True)

            # Update the velocity by augmenting it with the speed
            vel_augmented = torch.cat([vel, vel_norm], dim=1)
            return vel_augmented

        # Solve the ODE with the augmented state
        result = torchdiffeq.odeint(vel_len, x_augmented, self.t, method=self.method)[-1]
        # Separate the final position and the total accumulated length
        final_position = result[:, :-1]
        total_length = result[:, -1]

        # Here total_length contains the integral of the path length
        # You can use total_length for regularization or further computation
        return final_position, total_length
 
class NeuralODE_SR(nn.Module):
    def __init__(self, velocity, method='rk4',t=torch.linspace(0.0,1.0,10)):
        super().__init__()
        self.velocity = velocity
        self.method = method
        self.t = t        
        
    def forward(self, x):
        return torchdiffeq.odeint(self.velocity, x, self.t, method=self.method)
    
    def inverse(self, x):
        return self.ode(self.velocity, x, reversed(self.t),method=self.method)
        
    def clone(self):
        # Ensure we create the new instance with required initial parameters
        new_instance = NeuralODE(self.velocity.clone(), self.method, self.t.clone().detach())

        # Since method and t are immutable, it's fine to assign directly, but shown here for clarity
        new_instance.method = self.method
        new_instance.t = self.t.clone().detach()
        
        return new_instance

class NeuralODE(NeuralODE_SR):
    def __init__(self, velocity, method='rk4',t=torch.linspace(0.0,1.0,10)):
        super().__init__(velocity)     
        
    def forward(self, x, t=-1):
        return super().forward(x)[t]
    
    def inverse(self, x, t=-1):
        return super().forward(x)[t]
    

# # Sorry for adding another neuralode class here, symmetric loss gives me trouble if I dont do it this way.
# # func.functional_call only calls the forward method, so this is a way to change the definition of forward method for me
# class NeuralODE_with_Inverse(nn.Module):
#     def __init__(self, velocity, method='rk4',t=torch.linspace(0.0,1.0,10)):
#         super().__init__()
#         self.velocity = velocity
#         self.method = method
#         self.t = t        
        
#     def forward_blah(self, x):
#         return torchdiffeq.odeint(self.velocity, x, self.t, method=self.method)[-1]
    
#     def inverse(self, x):
#         def inv(t,x):
#             return -self.velocity(t,x)
#         return torchdiffeq.odeint(inv, x, torch.flip(self.t, dims=[0]),method=self.method)[-1]
    
#     def set_forward_method(self, method_name):
#         if method_name == 'forward':
#             self.forward = self.forward_blah
#         elif method_name == 'inverse':
#             self.forward = self.inverse

#     def forward(self, *args, **kwargs):
#         return self.forward(*args, **kwargs)
    
# this class allows one to stack multiple transformations
class TransformationSequence(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.transforms = nn.ModuleList(args)
        
    def forward(self, x, *args):
        for transform in self.transforms:
            x = transform(x, *args)
        return x

    def inverse(self, x):
        for transform in reversed(self.transforms):
            x = transform.inverse(x)
        return x

    def __str__(self):
        s = "Sequential: "
        for transform in self.transforms:
            s += str(transform) + " -> "
        return s[:-3]

if __name__ == '__main__':
    #Example
    from domain import Domain
    domain = Domain(torch.tensor([0.0, 1.0, 0.0, 1.0]), torch.tensor([5,8]))
    vel = VelocityNN(domain.dim, [domain.dim+1, 32])
    trafo = NeuralODE(vel)    

    trafo = Rigid2d()
    trafo.theta.data = torch.tensor([3.1415/4])
    trafo.Q()
    # trafo.b.data = torch.tensor([[1.0, 2.0]])
    x = torch.tensor([[1.0, 0.0],[0.0, 1.0],[1.0, 1.0],[0.0,0.0]])
    y = trafo(x)

    import matplotlib.pyplot as plt
    plt.Figure()
    plt.plot(x[:,0].numpy(), x[:,1].numpy(), 'ro')

    plt.plot(y[:,0].detach().numpy(), y[:,1].detach().numpy(), 'bs')
    plt.axis('equal')
    plt.show()

    