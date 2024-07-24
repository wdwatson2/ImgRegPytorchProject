import os
import numpy as np
import torch
import torch
import torchdiffeq
from PIL import Image
from src.domain import Domain
from src.utils import *
from src.super_resolution_tools import getRandomAffine
from src.transformations import Affine2d
from src.interpolation import SplineInter 

dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir('/'.join(dir_path.split('/')[:-1]))

# Get Images
print(100*'~')
print("Getting Images")
print(100*'~')

domain = Domain.load(os.getcwd() + '/data/dynamics/domain')
mx,my = domain.m

def im_to_tensor(im):
    return torch.fliplr(torch.tensor(im.getdata(), dtype=torch.float32).view(my, mx).transpose(0,1))

kidney_im = Image.open(os.getcwd() + '/data/dynamics/reference.png')
kidney_2d = im_to_tensor(kidney_im)

masks_2d = torch.stack([
    im_to_tensor(Image.open(os.getcwd() + '/data/dynamics/masks/medulla.png')),
    im_to_tensor(Image.open(os.getcwd() + '/data/dynamics/masks/cortex.png'))
])
masks_2d = 1.*(masks_2d>0)

# Simulate Physics
print(100*'~')
print("Simulating Physics")
print(100*'~')

time = torch.arange(0,60,2.4)
n_images = len(time)

# "True" physics
def AIF(t):
    # Create a mask for valid t values (greater than zero)
    valid_t = t > 0
    # Initialize result tensor filled with zeros
    result = torch.zeros_like(t)
    # Compute the AIF only for valid t values
    result[valid_t] = 4 * t[valid_t]**.1 * torch.exp(-(t[valid_t] - 50)/50)**2 + (20 * t[valid_t])/(t[valid_t] + 1) * torch.exp(1/(t[valid_t] + 1)) + 1 * torch.sin(t[valid_t]/10)
    
    return result

def get_dQ(AIF, params):
    dQ_P = lambda t, q : (params['F_T']/(1 - params['h'])) * AIF(t - params['delta']) * (t > params['delta']) - ((params['PS'] + params['F_T'])/(params['V_b'] * (1 - params['h'])))*q[0] + (params['PS'] / params['V_e'])*q[1]
    dQ_I = lambda t, q : (params['PS']/(params['V_b'] * (1-params['h']))) * q[0] - (params['PS']/params['V_e']) * q[1]

    return lambda t,q : torch.stack([dQ_P(t,q), dQ_I(t,q)],dim=0)

# Some parameters based on [Citation]
true_params = {
    'F_T'   : torch.tensor([20., 30.]), # [1/100 minutes]
    'V_b'   : torch.tensor([.40, .45]), # [%]
    'PS'    : torch.tensor([7., 12.]), # [1/100 minutes]
    'V_e'   : torch.tensor([.15, .30]), # [%]
    'delta' : 4.5, # [s]
    'h'     : .4
}

dQ = get_dQ(AIF, true_params)
Q0 = torch.zeros(2,2)

# Simulate stiff ODE system
Q = torchdiffeq.odeint(dQ, Q0, time)
alpha_true = torch.sum(Q,dim=1)


# Create true references
u_true = torch.einsum('ij,jkl->ikl' , alpha_true, masks_2d)
f = u_true + kidney_2d


# Add some motion
print(100*'~')
print("Adding Affine Transformations")
print(100*'~')
finterp = [SplineInter(f[j], domain, regularizer='moments', theta=0) for j in range(n_images)]

xc = domain.getCellCenteredGrid()
torch.manual_seed(42)
trafos = []
w = []
for j in range(n_images - 1):
    trafos.append(Affine2d())
    A, b = getRandomAffine(rotation_range=(-6,6), scale_range=(.9,1.1), seed=torch.randint(10000, (1,)).item())
    wc = torch.hstack([A.flatten(), b.flatten()])
    vec_to_params(trafos[j], wc)
    w.append(wc)
w = torch.hstack(w)

d = f.clone()
for j in range(n_images - 1):
    d[j + 1] = finterp[j](trafos[j](xc)).reshape(mx,my)

xmin=13
xmax=228
ymin=20
ymax=170


root_directory = '/data/dynamics/d_affine/'
print(100*'~')
print("Saving Synthetic Data to "+root_directory)
print(100*'~')
smaller_domain = Domain(torch.tensor([xmin, xmax, ymin, ymax]),torch.tensor([xmax-xmin, ymax-ymin]))
smaller_domain.save(os.getcwd() + root_directory + 'domain')
torch.save(time, os.getcwd() + root_directory + 'time.pt')
torch.save(d[:, xmin:xmax, ymin:ymax], os.getcwd() + root_directory+'d.pt')
torch.save(f[:, xmin:xmax, ymin:ymax], os.getcwd() + root_directory+'f_true.pt')
torch.save(masks_2d[:, xmin:xmax, ymin:ymax], os.getcwd() + root_directory+'masks.pt')
torch.save(w, os.getcwd() + root_directory + 'w_true.pt')
torch.save(alpha_true, os.getcwd() + root_directory + 'alpha_true.pt')