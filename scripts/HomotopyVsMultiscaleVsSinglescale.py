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
import time
import pandas as pd
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

def res(wc,theta):
    wp = unflatten_params(wc,keys,shapes,sizes)
    Timg = SplineInter(T, domain,regularizer='moments',theta=theta)
    yc = func.functional_call(trafo,wp,xc)
    Ty = Timg(yc)
    Rimg = SplineInter(R, domain,regularizer='moments',theta=theta)
    Rc = Rimg(xc)
    res = (Ty-Rc).squeeze(1)
    return torch.sqrt(torch.prod(domain.h))*res

def jac(wc,theta):
    jac = func.jacfwd(res)(wc,theta).squeeze(1)
    return jac

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

def singlescale(scaling, wc_np, track_loss=False):
    if not track_loss:
        theta = scaling[-1]
        theta_np = tensor_to_numpy(theta)
        results = minimize(scipy_loss, x0=wc_np, 
                           args=theta_np, method='trust-ncg',
                           jac=scipy_grad, hess=scipy_hessian,
                           options={'gtol': 1e-7})
        return results.x
    
    else:
        loss_single = []
        def callback_single(xk):
            loss_single.append(last_loss)

        theta = scaling[-1]
        theta_np = tensor_to_numpy(theta)
        results = minimize(scipy_loss, x0=wc_np, 
                           args=theta_np, method='trust-ncg',
                           jac=scipy_grad, hess=scipy_hessian,
                           options={'gtol': 1e-7}, callback=callback_single)
        return results.x, loss_single
        
def homotopyopt(scaling, wc_np, track_loss=False):
    if not track_loss:
        theta = scaling[0]
        theta_np = tensor_to_numpy(theta)
        results = minimize(scipy_loss, x0=wc_np, args=theta_np, method='trust-ncg',
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

            results = minimize(scipy_loss, x0=wc_np, args=theta_np, method='trust-ncg',
                    jac=scipy_grad, hess=scipy_hessian,
                    options={'gtol': 1e-4})        
            
        return results.x
    else:
        losses = []
        grads = []
        loss_homo = []
        grads_homo = []
        def callback_homo(xk):
            loss_homo.append(last_loss)
            grads_homo.append(last_grad)

        theta = scaling[0]
        theta_np = tensor_to_numpy(theta)
        results = minimize(scipy_loss, x0=wc_np, args=theta_np, method='trust-ncg',
                    jac=scipy_grad, hess=scipy_hessian,
                    options={'gtol': 1e-4}, callback=callback_homo)
        wc_np = results.x
        losses.append(loss_homo)
        grads.append(grads_homo)
        for i in range(1, len(scaling)):
            loss_homo = []
            grads_homo = []
            def velocity_fn(theta, wc):
                grad_grad_wc_theta = torch.func.jacrev(torch.func.grad(lossfn, argnums=0), argnums=1)(wc, theta).detach()
                vel = torch.linalg.lstsq(numpy_to_tensor(results.hess), grad_grad_wc_theta).solution 
                return -vel
            
            wc = torchdiffeq.odeint(velocity_fn, numpy_to_tensor(results.x), torch.tensor([theta, scaling[i]]), method='euler')[-1]
            wc_np = tensor_to_numpy(wc)

            theta = scaling[i]
            theta_np = tensor_to_numpy(theta)

            results = minimize(scipy_loss, x0=wc_np, args=theta_np, method='trust-ncg',
                    jac=scipy_grad, hess=scipy_hessian,
                    options={'gtol': 1e-4}, callback=callback_homo)      
            losses.append(loss_homo)  
            grads.append(grads_homo)
            
        return results.x, losses, grads
    
def homotopy_with_reference(scaling, wc_np):
    losses = []
    grads = []
    loss_homo = []
    grads_homo = []

    loss_references = []
    grad_references = []
    def callback_homo(xk):
        loss_homo.append(last_loss)
        grads_homo.append(last_grad)

    # compute loss without prediction step
    with torch.no_grad():
        loss_ref = lossfn(numpy_to_tensor(wc_np), scaling[0]).detach()
        grad_ref = torch.func.grad(lossfn, argnums=0)(numpy_to_tensor(wc_np), scaling[0]).detach()
        loss_references.append(loss_ref.item())
        grad_references.append(torch.norm(grad_ref).item())

    # normal minimization
    theta = scaling[0]
    theta_np = tensor_to_numpy(theta)
    results = minimize(scipy_loss, x0=wc_np, args=theta_np, method='trust-ncg',
                jac=scipy_grad, hess=scipy_hessian,
                options={'gtol': 1e-4}, callback=callback_homo)
    wc_np = results.x
    losses.append(loss_homo)
    grads.append(grads_homo)
    for i in range(1, len(scaling)):

        # compute loss without prediction step
        with torch.no_grad():
            loss_ref = lossfn(numpy_to_tensor(wc_np), scaling[i]).detach()
            grad_ref = torch.func.grad(lossfn, argnums=0)(numpy_to_tensor(wc_np), scaling[i]).detach()
            loss_references.append(loss_ref.item())
            grad_references.append(torch.norm(grad_ref).item())

        # Normal predictor corrector homotopy method
        loss_homo = []
        grads_homo = []
        def velocity_fn(theta, wc):
            grad_grad_wc_theta = torch.func.jacrev(torch.func.grad(lossfn, argnums=0), argnums=1)(wc, theta).detach()
            vel = torch.linalg.lstsq(numpy_to_tensor(results.hess), grad_grad_wc_theta).solution 
            return -vel
        
        wc = torchdiffeq.odeint(velocity_fn, numpy_to_tensor(results.x), torch.tensor([theta, scaling[i]]), method='euler')[-1]
        wc_np = tensor_to_numpy(wc)

        theta = scaling[i]
        theta_np = tensor_to_numpy(theta)

        results = minimize(scipy_loss, x0=wc_np, args=theta_np, method='trust-ncg',
                jac=scipy_grad, hess=scipy_hessian,
                options={'gtol': 1e-4}, callback=callback_homo)      
        losses.append(loss_homo)  
        grads.append(grads_homo)
        
    return results.x, losses, grads, loss_references, grad_references

def homotopy_gn_with_ref(scaling, wc):
    losses = []
    grads = []
    loss_references = []
    grad_references = []
    theta = scaling[0].reshape(1)

    with torch.no_grad():
        loss = torch.norm(res(wc, theta.item()))**2
        grad = func.grad(lambda wc, theta: torch.norm(res(wc, theta))**2)(wc, theta)
        loss_references.append(loss.item())
        grad_references.append(torch.norm(grad).item())

    results, hess, loss_lma, grad_lma = lsq_lma(p=wc,
            function=res, 
            jac_function=jac,
            args=theta,
            gtol=1e-4,
            ptol=1e-50,
            max_iter=1000,
            return_loss_and_grad=True)
    wc = results[-1]
    losses.append(loss_lma)
    grads.append(grad_lma)
    for i in range(1, len(scaling)): 
        with torch.no_grad():
            loss = torch.norm(res(wc, theta.item()))**2
            grad = func.grad(lambda wc, theta: torch.norm(res(wc, theta))**2)(wc, theta)
            loss_references.append(loss.item())
            grad_references.append(torch.norm(grad).item())

        def velocity_fn(theta,wc):
            grad_wc_theta = func.jacrev(func.grad(lambda wc, theta: torch.norm(res(wc, theta))**2), argnums=1)(wc, theta)
            vel = torch.linalg.lstsq(hess, grad_wc_theta).solution
            return -vel
        wc = torchdiffeq.odeint(velocity_fn, wc, torch.tensor([theta.item(), scaling[i]]), method='euler')[-1]

        theta = scaling[i].reshape(1) 
        results, hess, loss_lma, grad_lma = lsq_lma(p=wc,
                function=res, 
                jac_function=jac,
                args=theta,
                gtol=1e-5,
                ptol=1e-50,
                max_iter=1000,
                return_loss_and_grad=True)
        wc = results[-1]
        losses.append(loss_lma)
        grads.append(grad_lma)

    return wc, losses, grads, loss_references, grad_references

if __name__ == '__main__':

    scaling = torch.logspace(3, -3, 20)

    Timg = SplineInter(T, domain,regularizer='moments',theta=scaling[-1])
    T0 = Timg(xc)
    Rimg = SplineInter(R, domain,regularizer='moments',theta=scaling[-1])
    R0 = Rimg(xc)

    wc = params_to_vec(trafo)
    wc_np = tensor_to_numpy(wc)

    # Is the prediction step helping? 
    # Does using gauss newton hessian approx work as well as using an exact hessian?
    wc_homo, losses_homo, grads_homo, loss_ref_homo, grad_ref_homo = homotopy_with_reference(scaling, wc_np)
    wc_gn, losses_gn, grads_gn, loss_ref_gn, grad_ref_gn = homotopy_gn_with_ref(scaling, wc)
    print(f"difference between two solutions: {torch.norm(numpy_to_tensor(wc_homo) - wc_gn)}")
    loss_diff_n = []
    grad_diff_n = []
    loss_diff_gn = []
    grad_diff_gn = []
    for i in range(len(scaling)): #relative loss difference and relative grad difference
        loss_diff_n.append((loss_ref_homo[i] - losses_homo[i][0])/loss_ref_homo[i])
        grad_diff_n.append((grad_ref_homo[i] - grads_homo[i][0])/grad_ref_homo[i])

        loss_diff_gn.append((loss_ref_gn[i] - losses_gn[i][0])/loss_ref_gn[i])
        grad_diff_gn.append((grad_ref_gn[i] - grads_gn[i][0])/grad_ref_gn[i])
    plt.figure(figsize=(5,6))
    plt.subplot(2,1,1)
    plt.scatter(scaling, loss_diff_n)
    plt.scatter(scaling, loss_diff_gn)
    plt.axhline(y=0, color='gray', linestyle='--') 
    plt.ylabel('Loss Difference', fontsize=14)
    plt.xlabel(r'$\theta$', fontsize=14)
    plt.xscale('log')
    plt.title('Loss Difference', fontsize=16)

    plt.subplot(2,1,2)
    plt.scatter(scaling, grad_diff_n)
    plt.scatter(scaling, grad_diff_gn)
    plt.axhline(y=0, color='gray', linestyle='--') 
    plt.ylabel('Gradient Norm Difference', fontsize=14)
    plt.xlabel(r'$\theta$', fontsize=14)
    plt.xscale('log')
    plt.title('Gradient Difference', fontsize=16)
    plt.subplots_adjust(hspace=0.5)  

    plt.savefig('./results/fig/LossandGradientDiffPlot.png')
    plt.show()
    
    # run all optimization strategies
    wc_single, losses_single = singlescale(scaling, wc_np, track_loss=True)
    wc_homo, losses_homo, grads_homo = homotopyopt(scaling, wc_np, track_loss=True)

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
    plt.title(r'$\mathcal{T}(\mathbf{\omega}))$', fontsize=14)
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
    plt.title(r'$\mathcal{T}(\mathbf{\omega}))$ and $\vec{y}(\mathbf{\omega}; \mathbf{w})$', fontsize=14)
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
    plt.savefig('./results/fig/singlescaleFailure.png')
    plt.show()

    