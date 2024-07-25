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

iteration = 0
first_loss = None
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
    loss = distance(Ty,Rc)
    return loss

def scipy_loss(wc, theta):
    global last_loss
    global iteration
    global first_loss
    wc_tensor = numpy_to_tensor(wc)
    theta_tensor = numpy_to_tensor(theta)
    loss = lossfn(wc_tensor, theta_tensor).detach()
    if iteration == 0:
        first_loss = loss.item()
    last_loss = loss.item()
    iteration+=1
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
     
def homotopy_with_reference(scaling, wc_np):
    losses = []
    grads = []
    loss_homo = []
    grads_homo = []

    loss_references = []
    grad_references = []
    def callback_homo(xk):
        # callback is funky because callback is not called before the first first_loss is overwritten with the second iteration
        if iteration == 2:
            loss_homo.append(first_loss)
            loss_homo.append(last_loss)
        else:
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
    results = minimize(scipy_loss, x0=wc_np, args=theta_np, method='trust-exact',
                jac=scipy_grad, hess=scipy_hessian,
                options={'gtol': 1e-4}, callback=callback_homo)
    wc_np = results.x
    losses.append(loss_homo)
    grads.append(grads_homo)
    global iteration
    iteration = 0
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

        results = minimize(scipy_loss, x0=wc_np, args=theta_np, method='trust-exact',
                jac=scipy_grad, hess=scipy_hessian,
                options={'gtol': 1e-4}, callback=callback_homo)      
        losses.append(loss_homo)  
        grads.append(grads_homo)
        iteration = 0
        
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

    scaling = torch.logspace(3, -3, 40)

    wc = params_to_vec(trafo)
    wc_np = tensor_to_numpy(wc)

    # Is the prediction step helping? 
    # Can GN be used with predicter-corrector homotopy method
    # Does using gauss newton hessian approx work as well as using an exact hessian?
    # The answer is NO, GN hessian works against us, exact hessian actually helps
    wc_homo, losses_homo, grads_homo, loss_ref_homo, grad_ref_homo = homotopy_with_reference(scaling, wc_np)
    wc_gn, losses_gn, grads_gn, loss_ref_gn, grad_ref_gn = homotopy_gn_with_ref(scaling, wc)
    # Both optimization schemes arrive at the same set of paramaters:
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

    print(loss_diff_n)
    print(loss_diff_gn)
    print(grad_diff_n)
    print(grad_diff_gn)
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.scatter(scaling, loss_diff_n, marker = '.', label="Newton", s=100)
    plt.scatter(scaling, loss_diff_gn, marker='*', label="Gauss Newton", s=75)
    plt.axhline(y=0, color='gray', linestyle='--') 
    plt.ylabel('Relative Difference', fontsize=14)
    plt.xlabel(r'$\theta$', fontsize=14)
    plt.xscale('log')
    plt.title('Relative Loss Difference', fontsize=16)

    plt.subplot(1,2,2)
    plt.scatter(scaling, grad_diff_n, marker = '.', label="Newton", s=100)
    plt.scatter(scaling, grad_diff_gn, marker = '*', label="Gauss Newton", s=75)
    plt.axhline(y=0, color='gray', linestyle='--') 
    plt.ylabel('Relative Difference', fontsize=14)
    plt.xlabel(r'$\theta$', fontsize=14)
    plt.xscale('log')
    plt.title('Relative Grad Norm Difference', fontsize=16)
    plt.subplots_adjust(hspace=0.5) 
    plt.tight_layout() 
    plt.legend(fontsize='large')

    plt.savefig('./results/figs/LossandGradientDiffPlot.png')
    plt.show()


    