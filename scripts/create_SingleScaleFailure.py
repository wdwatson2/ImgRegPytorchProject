# load hands-R.jpg and convert to pytorch tensor
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from src.domain import Domain
from src.utils import *
import torch.func as func
from src.transformations import Affine2d
from scipy.optimize import minimize
import torchdiffeq
from src.interpolation import SplineInter
from src.distance import SSDDistance
from src.plotting import view_image_2d, plot_grid_2d
from src.optimization import lsq_lma
torch.set_default_dtype(torch.float64)

R = Image.open('./data/hands-R.jpg')
m = 32
R = R.resize((m, m))
R = torch.fliplr(torch.tensor(R.getdata(), dtype=torch.float64).view(m,m).transpose(0,1))

T = Image.open('./data/hands-T.jpg')
T = T.resize((m, m))
T = torch.flipud(torch.tensor(T.getdata(), dtype=torch.float64).view(m,m).transpose(0,1))

domain = Domain(torch.tensor([0.0, 20.0 ,0.0, 25.0]), torch.tensor([m, m]))

distance = SSDDistance(domain)
trafo = Affine2d()
keys = [k for k, _ in trafo.named_parameters()]
_, shapes, sizes = flatten_params({k: v.detach() for k, v in trafo.named_parameters()})
xc = domain.getCellCenteredGrid().view(-1,2)

last_loss = None
last_grad = None

def lossfn(wc, theta):
    wp = unflatten_params(wc,keys,shapes,sizes)
    yc = func.functional_call(trafo,wp,xc)
    Timg = SplineInter(T, domain,regularizer='moments',theta=theta)
    Ty = Timg(yc)
    Rimg = SplineInter(R, domain,regularizer='moments',theta=theta)
    Rc = Rimg(xc)
    # weightnormsq = torch.linalg.norm(wc[6:])**2 # weight decay regularization
    # loss = distance(Ty,Rc) + lam * weightnormsq
    loss = distance(Ty,Rc)
    return loss

def scipy_loss(wc, theta):
    global last_loss
    global last_wc
    wc_tensor = numpy_to_tensor(wc)
    last_wc = wc_tensor.detach()
    theta_tensor = numpy_to_tensor(theta)
    loss = lossfn(wc_tensor, theta_tensor)
    last_loss = loss.item()
    return last_loss

def scipy_grad(wc, theta):
    global last_grad
    wc = numpy_to_tensor(wc)
    theta = numpy_to_tensor(theta)
    grad = torch.func.grad(lossfn, argnums=0)(wc, theta).detach()
    npgrad = tensor_to_numpy(grad)
    last_grad = torch.norm(grad).item()
    return npgrad

def scipy_hessian(wc, theta):
    wc = numpy_to_tensor(wc)
    theta = numpy_to_tensor(theta)
    hessian = torch.func.hessian(lossfn, argnums=0)(wc, theta).detach()
    hessian = .5 * (hessian + hessian.T)
    return tensor_to_numpy(hessian)

def singlescale(scaling, wc_np):
    theta = scaling[-1]
    theta_np = tensor_to_numpy(theta)
    results = minimize(scipy_loss, x0=wc_np, 
                        args=theta_np, method='trust-exact',
                        jac=scipy_grad, hess=scipy_hessian,
                        options={'gtol': 1e-7})
    return results.x
        
def predictcorrect(scaling, wc_np, track_loss=False):
    theta = scaling[0]
    theta_np = tensor_to_numpy(theta)
    results = minimize(scipy_loss, x0=wc_np, args=theta_np, method='trust-exact',
                jac=scipy_grad, hess=scipy_hessian,
                options={'gtol': 1e-4})
    wc_np = results.x
    for i in range(1, len(scaling)):
        def velocity_fn(theta, wc):
            grad_grad_wc_theta = torch.func.jacrev(torch.func.grad(lossfn, argnums=0), argnums=1)(wc, theta).detach()
            vel = torch.linalg.lstsq(numpy_to_tensor(results.hess), grad_grad_wc_theta).solution 
            return -vel
        
        wc = torchdiffeq.odeint(velocity_fn, numpy_to_tensor(results.x), torch.tensor([theta, scaling[i]]), method='euler')[-1]
        wc_np = tensor_to_numpy(wc)

        theta = scaling[i]
        theta_np = tensor_to_numpy(theta)

        results = minimize(scipy_loss, x0=wc_np, args=theta_np, method='trust-exact',
                jac=scipy_grad, hess=scipy_hessian,
                options={'gtol': 1e-4})        
        
    return results.x
   

if __name__ == '__main__':

    scaling = torch.logspace(3, -3, 30)

    Timg = SplineInter(T, domain,regularizer='moments',theta=scaling[-1])
    T0 = Timg(xc)
    Rimg = SplineInter(R, domain,regularizer='moments',theta=scaling[-1])
    R0 = Rimg(xc)

    wc = params_to_vec(trafo)
    wc_np = tensor_to_numpy(wc)
    
    # run singlescale and homotopy opt
    wc_single  = singlescale(scaling, wc_np)
    wc_homo = predictcorrect(scaling, wc_np)

    # Creation of single scale failing plot
    trafo1 = Affine2d()
    trafo2 = Affine2d()
    vec_to_params(trafo1, numpy_to_tensor(wc_homo))
    vec_to_params(trafo2, numpy_to_tensor(wc_single))
    residual1 = (T0 - R0)
    residual2 = (Timg(trafo1(xc).detach()) - R0)
    residual3 = (Timg(trafo2(xc).detach()) - R0)
    vmin = min(torch.min(a=residual1), torch.min(residual2), torch.min(residual3))
    vmax = max(torch.max(residual1), torch.max(residual2), torch.min(residual3))
    plt.figure(figsize=(8,8))
    plt.subplot(3, 3, 1)
    view_image_2d(T0, domain)
    plt.title(r'$\mathcal{T}(\mathbf{\omega})$', fontsize=14)
    plt.subplot(3, 3, 2)
    view_image_2d(R0, domain)
    plt.title(r'$\mathcal{R}(\mathbf{\omega})$', fontsize=14)
    plt.subplot(3, 3, 3)
    view_image_2d(residual1, domain, kwargs = {'cmap': 'gray', 'vmin': vmin, 'vmax': vmax})
    plt.title(r'$\mathcal{T}(\mathbf{\omega}) - \mathcal{R}(\mathbf{\omega})$', fontsize=14)
    plt.colorbar()
    plt.subplot(3, 3, 4)
    view_image_2d(T0, domain)
    plot_grid_2d(trafo1(domain.getNodalGrid()).detach(), domain, spacing=4)
    plt.title(r'$\mathcal{T}(\mathbf{\omega})$ and $\vec{y}(\mathbf{\omega}; \mathbf{w})$', fontsize=14)
    plt.subplot(3, 3, 5)
    view_image_2d(Timg(trafo1(xc).detach()), domain)
    plt.title(r'$\mathcal{T}(\vec{y}(\mathbf{\omega}; \mathbf{w}))$', fontsize=14)
    plt.subplot(3, 3, 6)
    view_image_2d(residual2, domain,kwargs = {'cmap': 'gray', 'vmin': vmin, 'vmax': vmax})
    plt.title(r'$\mathcal{T}(\vec{y}(\mathbf{\omega}; \mathbf{w})) - \mathcal{R}(\mathbf{\omega})$', fontsize=14)
    plt.colorbar()
    plt.subplot(3, 3, 7)
    view_image_2d(T0, domain)
    plot_grid_2d(trafo2(domain.getNodalGrid()).detach(), domain, spacing=4)
    # plt.title(r'$\mathcal{T}(\mathbf{\omega}))$ and $\vec{y}(\mathbf{\omega}; \mathbf{w})$', fontsize=14)
    plt.subplot(3, 3, 8)
    view_image_2d(Timg(trafo2(xc).detach()), domain)
    # plt.title(r'$\mathcal{T}(\vec{y}(\mathbf{\omega}; \mathbf{w}))$', fontsize=14)
    plt.subplot(3, 3, 9)
    view_image_2d(residual3, domain,kwargs = {'cmap': 'gray', 'vmin': vmin, 'vmax': vmax})
    # plt.title(r'$\mathcal{T}(\vec{y}(\mathbf{\omega}; \mathbf{w})) - \mathcal{R}(\mathbf{\omega})$', fontsize=14)
    plt.colorbar()
    plt.subplots_adjust(hspace=0.5)  # Adjust vertical spacing
    plt.tight_layout()
    plt.savefig('./results/figs/singlescaleFailure.png')
    plt.show()

    