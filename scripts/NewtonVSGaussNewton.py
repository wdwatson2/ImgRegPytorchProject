# load hands-R.jpg and convert to pytorch tensor
import torch
from PIL import Image
import matplotlib.pyplot as plt
from src.domain import Domain
import time
from src.plotting import view_image_2d, plot_grid_2d
from src.utils import *
import numpy as np
import torch.func as func
from src.transformations import Affine2d
from src.distance import SSDDistance
from src.interpolation import SplineInter
torch.set_default_dtype(torch.float64)

R = Image.open('./data/hands-R.jpg')
m = 32
R = R.resize((m, m))
R = torch.fliplr(torch.tensor(R.getdata(), dtype=torch.float32).view(m,m).transpose(0,1))

T = Image.open('./data/hands-T.jpg')
T = T.resize((m, m))
T = torch.fliplr(torch.tensor(T.getdata(), dtype=torch.float32).view(m,m).transpose(0,1))

domain = Domain(torch.tensor([0.0, 20.0 ,0.0, 25.0]), torch.tensor([m, m]))
xc = domain.getCellCenteredGrid().view(-1,2)
theta = torch.tensor(1e2) 
distance = SSDDistance(domain)
trafo = Affine2d()

Timg = SplineInter(T, domain,regularizer='moments',theta=theta)
Rimg = SplineInter(R, domain,regularizer='moments',theta=theta)
R0 = Rimg(xc)
T0 = Timg(xc)

keys = [k for k, _ in trafo.named_parameters()]
_, shapes, sizes = flatten_params({k: v.detach() for k, v in trafo.named_parameters()})
wc = params_to_vec(trafo)

def Ty(wc,xc):
    wp = unflatten_params(wc,keys,shapes,sizes)
    yc = func.functional_call(trafo,wp,xc)
    Ty = Timg(yc)
    return Ty, Ty

def lossfn(wc):
    wp = unflatten_params(wc,keys,shapes,sizes)
    yc = func.functional_call(trafo,wp,xc)
    Ty = Timg(yc)
    return distance(Ty,R0)

# Stopping criteria
gtol = 1e-4

if __name__ == '__main__':

    # Gauss Newton with simple backtracking linesearch
    gn_loss = []
    gn_grads = []
    for iter in range(100):
        Jac_fwd, Tyc = func.jacfwd(Ty, has_aux=True)(wc, xc)
        Jac_fwd = Jac_fwd.squeeze(1)
        res_fwd = Tyc-R0
        loss  = distance(Tyc,R0) 
        gn_loss.append(loss.item())
        grad = Jac_fwd.T@res_fwd 
        norm_grad = torch.norm(grad)
        gn_grads.append(norm_grad.item())

        if norm_grad < gtol:
            break

        s = torch.linalg.lstsq(Jac_fwd, res_fwd).solution.squeeze(1)
        
        alpha = 1.
        for _ in range(10):
            wp = wc - alpha * s
            Tyt = Ty(wp, xc)[0]
            newloss = distance(Tyt, R0)
            if newloss < loss:
                wc = wp
                break
            alpha *= 0.5

    trafo = Affine2d()
    wc = params_to_vec(trafo)

    # Pure Newton with same simple backtracking linesearch
    n_grads = []
    n_loss = []
    for iter in range(100):
        loss = lossfn(wc)
        n_loss.append(loss.item())
        grad = torch.func.grad(lossfn)(wc).detach()
        hessian = torch.func.jacrev(torch.func.grad(lossfn))(wc).detach()
        hessian = .5 * (hessian + hessian.T)
        norm_grad = torch.norm(grad)
        n_grads.append(norm_grad.item())

        if norm_grad < gtol:
            break

        s = torch.linalg.solve(hessian, grad)
        
        alpha = 1.
        for _ in range(10):
            wp = wc - alpha * s
            Tyt = Ty(wp, xc)[0]
            newloss = distance(Tyt, R0)
            if newloss < loss:
                wc = wp
                break
            alpha *= 0.5

    # Creation of gradient norm plot in paper
    plt.figure(figsize=(4,6))
    plt.plot(range(len(n_grads)), n_grads, label="Newton", marker='.', markersize=10)
    plt.plot(range(len(gn_grads)), gn_grads, label="Gauss Newton", marker='*', markersize=10)
    plt.ylabel(r"$\|\nabla J\|$", fontsize=14, labelpad=5)
    plt.xlabel("Iteration", fontsize=14)
    plt.yscale('log')
    plt.title('Gradient Norm Plot', fontsize=16)
    plt.legend()
    plt.tight_layout()
    plt.savefig('./results/figs/gradient_plot.png')
    plt.show()

    # Creation of N vs GN plot in paper
    vec_to_params(trafo, wc)
    residual1 = (T0 - R0)
    residual2 = (Timg(trafo(xc).detach()) - R0)
    vmin = min(torch.min(a=residual1), torch.min(residual2))
    vmax = max(torch.max(residual1), torch.max(residual2))
    plt.figure(figsize=(8,6))
    plt.subplot(2, 3, 1)
    view_image_2d(T0, domain)
    plt.title(r'$\mathcal{T}(\mathbf{\omega}))$', fontsize=14)
    plt.subplot(2, 3, 2)
    view_image_2d(R0, domain)
    plt.title(r'$\mathcal{R}(\mathbf{\omega})$', fontsize=14)
    plt.subplot(2, 3, 3)
    view_image_2d(residual1, domain, kwargs = {'cmap': 'gray', 'vmin': vmin, 'vmax': vmax})
    plt.title(r'$\mathcal{T}(\mathbf{\omega}) - \mathcal{R}(\mathbf{\omega})$', fontsize=14)
    plt.colorbar()
    plt.subplot(2, 3, 4)
    view_image_2d(T0, domain)
    plot_grid_2d(trafo(domain.getNodalGrid()).detach(), domain, spacing=4)
    plt.title(r'$\mathcal{T}(\mathbf{\omega}))$ and $\vec{y}(\mathbf{\omega}; \mathbf{w})$', fontsize=14)
    plt.subplot(2, 3, 5)
    view_image_2d(Timg(trafo(xc).detach()), domain)
    plt.title(r'$\mathcal{T}(\vec{y}(\mathbf{\omega}; \mathbf{w}))$', fontsize=14)
    plt.subplot(2, 3, 6)
    view_image_2d(residual2, domain,kwargs = {'cmap': 'gray', 'vmin': vmin, 'vmax': vmax})
    plt.title(r'$\mathcal{T}(\vec{y}(\mathbf{\omega}; \mathbf{w})) - \mathcal{R}(\mathbf{\omega})$', fontsize=14)
    plt.colorbar()
    plt.subplots_adjust(hspace=0.5)  # Adjust vertical spacing
    plt.tight_layout()
    plt.savefig('./results/figs/results_single_scale.png')
    plt.show()




