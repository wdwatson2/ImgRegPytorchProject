#####################################################################################################
# Creates perc CG VARPRO figures; run from the repository directory, not scripts                    #
#####################################################################################################
# WARNING: Due to issues with PyTorch's forward mode automatic differentiation,                     #
#          this is a very memory intensive script for even modest numbers and                       #
#          sizes of images. This should be rewritten in jax for future applications.                #                                                              
#####################################################################################################



import torch
from PIL import Image
import matplotlib.pyplot as plt
from src.domain import Domain
from src.interpolation import SplineInter
from src.transformations import Affine2d
from src.plotting import view_image_2d, plot_grid_2d
from src.super_resolution_tools import *
from src.transformations import Affine2d
from src.LinearOperator import LinearOperator
from src.lanczos import lanczos_tridiag
from src.utils import * 
from src.optimization import *
import torch.func as func
from src.distance import SSDDistance
import time

torch.set_default_dtype(torch.float64)



def show_reference_and_templates(R, T):

    # Determine the layout for the subplots
    num_images = 1 + T.shape[0]  # Total number of images (reference + targets)
    cols = int(torch.ceil(torch.sqrt(torch.tensor(num_images).float())))  # Number of columns (and rows in a square layout)
    rows = int(torch.ceil(torch.tensor(num_images).float() / cols))  # Number of rows needed to accommodate all images

    # Create the figure with subplots
    _, ax = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))

    # Flatten the ax array for easier indexing
    ax = ax.flatten()

    # Display the reference image
    ax[0].imshow(R.detach().T.flip(dims=(0,)), cmap='gray')  # Convert tensor to NumPy array for imshow
    ax[0].set_title('Reference')
    ax[0].axis('off')  # Optional: hide axes for cleaner visualization

    # Display the target images
    for k in range(T.shape[0]):
        ax[k + 1].imshow(T[k].detach().T.flip(dims=(0,)), cmap='gray')  # Convert tensor to NumPy array for imshow
        ax[k + 1].set_title(f'T_{k}')
        ax[k + 1].axis('off')  # Optional: hide axes for cleaner visualization

    # Turn off any unused subplots
    for i in range(1 + T.shape[0], len(ax)):
        ax[i].axis('off')

    plt.tight_layout()  # Adjust subplots to fit into the figure area nicely
    #plt.show()




m = 8
factor = 2
theta = 0
n_images = 1

print(80*'-')
print('LOADING IMAGES')
print('m = {}'.format(m))
print('factor = {}'.format(factor))
print('theta = {}'.format(theta))
print('n_images = {}'.format(n_images))
print('Will attempt to reconstruct an {} x {} image from {} {}x{} images'.format(
    m, m, n_images, m//factor, m//factor
))
print(80*'-')

domain_R = Domain(torch.tensor((0, 20, 0, 25)), torch.tensor((m, m)))
domain_R.m.detach()
domain_T = Domain(torch.tensor((0, 20, 0, 25)), torch.tensor((m//factor, m//factor)))
xc = domain_R.getCellCenteredGrid().view(-1, 2).to(torch.float64)
times = torch.linspace(0, 15, n_images)

R = Image.open('data/hands-R.jpg')
R = R.resize((m, m))
R = torch.fliplr(torch.tensor(R.getdata(), dtype=torch.float64).view(m,m).transpose(0,1))
Rimg = SplineInter(R, domain_R ,regularizer='moments',theta=theta).to(torch.float64)

def randomAffines(reference, n):
    affines = [getRandomAffine(rotation_range=(0,45),seed=_+2) for _ in range(n-1)]
    templates = [down_sample(reference(xc).reshape(m, m), factor)]
    for aff in affines:
        affine = Affine2d()
        affine.A = torch.nn.Parameter(aff[0].to(torch.float64))
        affine.b = torch.nn.Parameter(aff[1].to(torch.float64))
        templates.append(down_sample(reference(affine(xc)).reshape(m, m), factor))
    return torch.stack(templates)

T = randomAffines(Rimg, len(times))

#show_reference_and_templates(R, T)

print(80*'-')
print('COMPLETED LOADING IMAGES')
print(80*'-')



print(80*'-')
print('PLOTTING SUPER RESOLUTION')
print(80*'-')

fig, ax = plt.subplots(1, n_images + 1, figsize = (10,2.6))

[a.axis('off') for a in ax]

plt.sca(ax[0])
view_image_2d(R, domain_R)
plt.title("True Reference")

for i in range(n_images):
    plt.sca(ax[i+1])
    view_image_2d(T[i].detach(), domain_T)
    plt.title("Template {}".format(i+1))

plt.suptitle("Test Super Resolution Problem", fontsize=18, y=1.07)

plt.savefig('results/figs/SR_test.pdf', bbox_inches='tight')

#plt.show()

print(80*'-')
print('COMPLETED PLOTTING SUPER RESOLUTION')
print(80*'-')



def L_forward(f):
    reference = f.reshape(*domain_R.m)
    diff_0 = torch.diff(reference, dim = 0) / domain_R.h[0]
    diff_1 = torch.diff(reference, dim = 1) / domain_R.h[1]
    return torch.hstack([diff_0.flatten(), diff_1.flatten()])

L = LinearOperator((torch.prod(domain_R.m).item(),), forward_fun=L_forward, dtype = torch.float64)

def K_forward(f):
    reference = f.reshape(*domain_R.m)
    downsampled = down_sample(reference, factor).flatten()
    return downsampled.flatten() 

K = LinearOperator((torch.prod(domain_R.m).item(),), forward_fun=K_forward, dtype = torch.float64)
# f0_interp = K.T @ T[0].flatten().detach()
Timg = SplineInter(T[0].detach(),domain_T, regularizer='moments', theta=1e-2)
f0_interp = Timg(domain_R.getCellCenteredGrid()).flatten()

d = T.flatten()
b = torch.hstack([d * torch.sqrt(torch.prod(domain_T.h)), torch.zeros_like(L @ f0_interp)])




xc_2d = domain_R.getCellCenteredGrid()
xc = xc_2d.reshape(torch.prod(domain_R.m).item(),2).detach()

ys = [Affine2d() for _ in range(n_images)]
for y in ys:
    for param in y.parameters():
        param.to(torch.float64)
        param.requires_grad_(True)

lam = 1e-2

wps = [{k: v for k, v in y.named_parameters()} for y in ys]
keys_list = [wp.keys() for wp in wps]

flat_params_list, shapes_list, sizes_list = flatten_params_list(wps)
wp_vec0 = torch.stack(flat_params_list).flatten().unsqueeze(1)






f0_grad = None
f0_nograd = None
def conjugate_gradient(A, b, x0, max_iter=500, tol=1e-3):
    x = x0.clone()
    r = b - A(x)
    p = r.clone()
    rsold = r.dot(r)
        
    for i in range(max_iter):
        Ap = A(p)
        alpha = rsold / p.dot(Ap)
        x = x + alpha * p  
        r = r - alpha * Ap  
        rsnew = r.dot(r)
        if torch.sqrt(rsnew) < tol:
            # print("early iters",i)
            # print(f"tol reached {i}")
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew
    # print(f"residual norm: {torch.sqrt(rsnew)}")
    # print("iters", i)
    return x
    

def conjugate_gradient_nograd(A, b, x0, max_iter=500, tol=1e-3):
    with torch.no_grad():
        x = x0.clone()
        r = b - A(x)
        p = r.clone()
        rsold = r.dot(r)
        for i in range(max_iter):
            Ap = A(p)
            alpha = rsold / p.dot(Ap)
            x = x + alpha * p 
            x = x.detach()
            r = r - alpha * Ap  
            rsnew = r.dot(r)
            if torch.sqrt(rsnew) < tol:
                # print("early iters",i)
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew
        # print(f"residual norm: {torch.sqrt(rsnew)}")
        # print("iters", i)
        return x

def Forward_single(wp, y, f_inter):
    w_dict = unflatten_params(wp, keys_list[0], shapes_list[0], sizes_list[0])
    yc = func.functional_call(y, w_dict, xc)
    d_pred = K(f_inter(yc))
    return d_pred

def res_fn(wp_vec, cg_iter):
    def A_forward(f):
        reference = f.reshape(*domain_R.m)        
        reference_img = SplineInter(reference, domain_R).to(torch.float64)
        template_predictions = torch.hstack([Forward_single(wp_vec.reshape(-1,6)[j], ys[j], reference_img)[0] for j in range(n_images)])

        regularizer = lam * L(f)
        
        return torch.hstack([template_predictions * torch.sqrt(torch.prod(domain_T.h)),
                                regularizer * torch.sqrt(torch.prod(domain_R.h))])
    
    A = LinearOperator((torch.prod(domain_R.m).item(),), forward_fun = A_forward, dtype = torch.float64)

    f0 = conjugate_gradient(A.T ^ A, A.T @ b, f0_interp, max_iter=cg_iter)
    reference = f0.reshape(*domain_R.m)        
    reference_img = SplineInter(reference, domain_R).to(torch.float64)
    template_predictions = torch.hstack([Forward_single(wp_vec.reshape(-1,6)[j], ys[j], reference_img)[0] for j in range(n_images)])

    return template_predictions - d

def res_fn_nodiff(wp_vec, cg_iter):
    def A_forward(f):
        reference = f.reshape(*domain_R.m)        
        reference_img = SplineInter(reference, domain_R)
        template_predictions = torch.hstack([Forward_single(wp_vec.reshape(-1,6)[j], ys[j], reference_img)[0] for j in range(n_images)])

        regularizer = lam * L(f)
        
        return torch.hstack([template_predictions * torch.sqrt(torch.prod(domain_T.h)),
                                regularizer * torch.sqrt(torch.prod(domain_R.h))])
    
    A = LinearOperator((torch.prod(domain_R.m).item(),), forward_fun = A_forward, dtype = torch.float64)

    f0 = conjugate_gradient_nograd(A.T ^ A, A.T @ b, f0_interp, max_iter= cg_iter)
    reference = f0.reshape(*domain_R.m)        
    reference_img = SplineInter(reference, domain_R)
    template_predictions = torch.hstack([Forward_single(wp_vec.reshape(-1,6)[j], ys[j], reference_img)[0] for j in range(n_images)])

    return template_predictions - d

def res_fn_lastdiff(wp_vec, cg_iter, somegrad_percent):
    def A_forward(f):
        reference = f.reshape(*domain_R.m)        
        reference_img = SplineInter(reference, domain_R)
        template_predictions = torch.hstack([Forward_single(wp_vec.reshape(-1,6)[j], ys[j], reference_img)[0] for j in range(n_images)])

        regularizer = lam * L(f)
        
        return torch.hstack([template_predictions * torch.sqrt(torch.prod(domain_T.h)),
                                regularizer * torch.sqrt(torch.prod(domain_R.h))])
    
    A = LinearOperator((torch.prod(domain_R.m).item(),), forward_fun = A_forward, dtype = torch.float64)

    f0_early = conjugate_gradient_nograd(A.T ^ A, A.T @ b, f0_interp, int(cg_iter * (1 - somegrad_percent)))
    f0 = conjugate_gradient(A.T ^ A, A.T @ b, f0_early, int(cg_iter * somegrad_percent))
    reference = f0.reshape(*domain_R.m)        
    reference_img = SplineInter(reference, domain_R)
    template_predictions = torch.hstack([Forward_single(wp_vec.reshape(-1,6)[j], ys[j], reference_img)[0] for j in range(n_images)])

    return template_predictions - d

def res_fn_firstdiff(wp_vec, cg_iter, somegrad_percent):
    def A_forward(f):
        reference = f.reshape(*domain_R.m)        
        reference_img = SplineInter(reference, domain_R)
        template_predictions = torch.hstack([Forward_single(wp_vec.reshape(-1,6)[j], ys[j], reference_img)[0] for j in range(n_images)])

        regularizer = lam * L(f)
        
        return torch.hstack([template_predictions * torch.sqrt(torch.prod(domain_T.h)),
                                regularizer * torch.sqrt(torch.prod(domain_R.h))])
    
    A = LinearOperator((torch.prod(domain_R.m).item(),), forward_fun = A_forward, dtype = torch.float64)

    f0_early = conjugate_gradient(A.T ^ A, A.T @ b, f0_interp, int(cg_iter * somegrad_percent))
    f0 = conjugate_gradient_nograd(A.T ^ A, A.T @ b, f0_early, int(cg_iter * (1 - somegrad_percent)))
    reference = f0.reshape(*domain_R.m)        
    reference_img = SplineInter(reference, domain_R)
    template_predictions = torch.hstack([Forward_single(wp_vec.reshape(-1,6)[j], ys[j], reference_img)[0] for j in range(n_images)])

    return template_predictions - d

def Jac_fn(wp_vec, cg_iter):
    return func.jacfwd(res_fn)(wp_vec, cg_iter).detach().squeeze()

def Jac_fn_nodiff(wp_vec, cg_iter):
    return func.jacfwd(res_fn_nodiff)(wp_vec, cg_iter).detach().squeeze()

def Jac_fn_lastdiff(wp_vec, cg_iter, somegrad_percent):
    return func.jacfwd(res_fn_lastdiff)(wp_vec, cg_iter, somegrad_percent).detach().squeeze()

def Jac_fn_firstdiff(wp_vec, cg_iter, somegrad_percent):
    return func.jacfwd(res_fn_firstdiff)(wp_vec, cg_iter, somegrad_percent).detach().squeeze()










# preregistration

from scipy.optimize import minimize

def single_img_trafo(wp,y):
    w_dict = unflatten_params(wp, keys_list[0], shapes_list[0], sizes_list[0])
    yc = func.functional_call(y, w_dict, domain_T.getCellCenteredGrid())
    # print(Timg(yc).shape)
    return Timg(yc).flatten()

def lossfn(wp_vec):
    pred_d = torch.hstack([single_img_trafo(wp_vec.reshape(-1,6)[j], ys[j]) for j in range(n_images)])
    return torch.norm(pred_d - d)**2


def scipy_loss(wc):
    global last_loss
    wc_tensor = numpy_to_tensor(wc)
    loss = lossfn(wc_tensor)
    last_loss = loss.item()
    return last_loss

def scipy_grad(wc):
    global last_grad
    wc = numpy_to_tensor(wc)
    grad = torch.func.grad(lossfn, argnums=0)(wc).detach()
    last_grad = tensor_to_numpy(grad)
    return last_grad

def scipy_hessian(wc):
    wc = numpy_to_tensor(wc)
    hessian = torch.func.hessian(lossfn, argnums=0)(wc).detach()
    hessian = .5 * (hessian + hessian.T)
    return tensor_to_numpy(hessian)

def callback(xk):
    norm_grad = np.linalg.norm(last_grad)
    grad_norms.append(norm_grad)
    global iteration_num
    iteration_num+=1
    print(f'{iteration_num} - Current Loss: {last_loss:.10e}, Norm of Gradient: {norm_grad:.10e}')

# multi-scale 


gtol = 1e-2
max_iter = 50

print(80*'-')
print('INITIATING PREREGISTRATION WITH NEWTON\'S METHOD')
print('gtol = {:.2e}'.format(gtol))
print('max_iter = {}'.format(max_iter))
print(80*'-')

thetas = torch.logspace(2, -3, 6)

for theta in thetas:
    print(80*'~')
    print('PREREGISTERING AT SCALE θ = {}'.format(theta))
    print(80*'~')

    theta_np = tensor_to_numpy(theta)

    last_loss = None
    last_grad = None
    iteration_num = 0

    grad_norms = []

    Timg = SplineInter(T[0].detach(), domain_T,regularizer='moments',theta=theta)
    Rimgs = [SplineInter(T[i], domain_T, regularizer='moments', theta=theta) for i in range(n_images)]
    Rcs = torch.stack([Rimgs[i](domain_T.getCellCenteredGrid()) for i in range(len(Rimgs))])

    results = minimize(scipy_loss, x0=tensor_to_numpy(wp_vec0).flatten(), method='newton-cg',
               jac=scipy_grad, hess=scipy_hessian,
               options={'gtol': gtol, 'maxiter':max_iter}, callback=callback)
    
    template_predictions = torch.stack([single_img_trafo(numpy_to_tensor(results.x).reshape(-1,6)[j], ys[j]).reshape(*domain_T.m) for j in range(n_images)])
    #show_reference_and_templates(T[0], template_predictions)

    print(80*'~')
    print('PREREGISTRATION COMPLETE AT SCALE θ = {}'.format(theta))
    print(80*'~')

print(80*'-')
print('PREREGISTRATION WITH NEWTON\'S METHOD COMPLETE')
print(80*'-')




maxcg_iters = 200 # verified that all iterations reach max_iters before stopping
proportions = [0.7,  0.9, 0.95, 1]
max_iter = 10

f0_grads = []
loss_grads = []
grad_grads = []

print(80*'-')
print('INITIATING \'percCG\' TEST')
print(80*'-')

for prop in proportions:
    print(80*'-')
    print('INITIATION \'percCG\' TEST AT PROPORTION {}'.format(prop))
    print(80*'-')


    wp_list,_,losses,grads = lsq_lma(numpy_to_tensor(results.x).flatten(), 
                                     res_fn_lastdiff, 
                                     Jac_fn_lastdiff, 
                                     args=[maxcg_iters, prop], 
                                     gtol=0, max_iter=max_iter, 
                                     verbose=True, 
                                     return_loss_and_grad=True)
    wp_vec = wp_list[-1]
    def A_forward(f):
        reference = f.reshape(*domain_R.m)        
        reference_img = SplineInter(reference, domain_R)
        template_predictions = torch.hstack([Forward_single(wp_vec.reshape(-1,6)[j], ys[j], reference_img)[0] for j in range(n_images)])
        regularizer = lam * L(f)
        return torch.hstack([template_predictions * torch.sqrt(torch.prod(domain_T.h)),
                                regularizer * torch.sqrt(torch.prod(domain_R.h))])
    A = LinearOperator((torch.prod(domain_R.m).item(),), forward_fun = A_forward, dtype = torch.float64)
    f0 = conjugate_gradient_nograd(A.T ^ A, A.T @ b, f0_interp, max_iter= 5000, tol=1e-12)
    f0_grad_reshaped = f0.reshape(*domain_R.m)

    f0_grads.append(f0_grad_reshaped.clone().detach())
    loss_grads.append(losses)
    grad_grads.append(grads)

    print(80*'-')
    print('COMPLETED \'percCG\' TEST AT PROPORTION {}'.format(prop))
    print(80*'-')


print(80*'-')
print('COMPLETED \'percCG\' TEST')
print(80*'-')



print(80*'-')
print('PLOTTING \'percCG\' TEST')
print(80*'-')

fig, ax = plt.subplots(1, 3, figsize = (15, 3))

plt.sca(ax[0])
for i in range(max_iter):
    plt.loglog(np.arange(max_iter) + 1, 1/(np.arange(max_iter) + 1)**(i), color='gray', alpha=.3, linestyle=':')
for i in range(len(proportions)):
    plt.loglog(np.arange(max_iter) + 1, np.array(loss_grads[i])/loss_grads[i][0], label="{}% CG iters".format(int(proportions[i]*100)), linewidth=0.5, marker = '.')
plt.legend()
plt.ylim(1e-4,1.5)
plt.title('Relative Loss')
plt.xlabel('Optimization Iterations')

plt.sca(ax[1])
for i in range(max_iter):
    plt.loglog(np.arange(max_iter) + 1, 1/(np.arange(max_iter) + 1)**(i), color='gray', alpha=.3, linestyle=':')
for i in range(len(proportions)):
    plt.loglog(np.arange(max_iter) + 1, np.array(grad_grads[i])/grad_grads[i][0], label="{}% CG iters".format(int(proportions[i]*100)), linewidth=0.5, marker = '.')
plt.legend()
plt.ylim(1e-4,10)
plt.title('Relative Gradient Norm')
plt.xlabel('Optimization Iterations')

plt.sca(ax[2])
plt.plot(proportions, np.array([torch.norm(f - R) for f in f0_grads])/torch.norm(R), linewidth=0.5, marker = '.', markersize=10, color='maroon')
plt.title('Final Relative Error')
plt.xlabel(f'Last CG Iters Diff (%)')

plt.suptitle('Performance When Differentiating Last CG Iters', y = 1.1, size=20)
plt.savefig('results/figs/performance_percentage.pdf', bbox_inches='tight')
#plt.show()



fig, ax = plt.subplots(2, len(proportions), figsize=(12,4))

for i in range(len(proportions)):
    #ax[0,i].axis('off')
    #ax[1,i].axis('off')

    plt.sca(ax[0,i])
    im_0 = view_image_2d(f0_grads[i],domain_R, kwargs={'clim':(0,250), 'cmap':'Greys_r'})
    plt.title("{}% CG Iters".format(int(proportions[i]*100)))

    plt.sca(ax[1,i])
    im_1 = view_image_2d(f0_grads[i] - R,domain_R, kwargs={'clim':(-50,50), 'cmap':'Greys_r'})

ax[0,0].set_ylabel("Reconstruction")
ax[1,0].set_ylabel("Reconstruction - True")

fig.subplots_adjust(0,0,.84,1)

# Create colorbar axes
cbar_ax1 = fig.add_axes([0.85, 0.55, 0.02, 0.45])  # x, y, width, height
cbar_ax2 = fig.add_axes([0.85, 0.0, 0.02, 0.45])

# Create colorbars in the new axes
cbar1 = plt.colorbar(im_0, cax=cbar_ax1)
cbar2 = plt.colorbar(im_1, cax=cbar_ax2)

plt.suptitle('Reconstructions When Differentiating Last CG Iters', y = 1.15, x=.45, size=20)

plt.savefig('results/figs/reconstructions_percentage.pdf', bbox_inches='tight')

plt.figure()

H = torch.logspace(5, -15, 10)
# Loop through each subplot and plot

seventy = []
ninety = []
ninety_five = []
all_grad = []
no_grad = []
v = Jac_fn_lastdiff(numpy_to_tensor(results.x).flatten(), maxcg_iters, 1).T @ res_fn_nodiff(numpy_to_tensor(results.x).flatten(), maxcg_iters)
v /= torch.norm(v)
f_x = res_fn_nodiff(numpy_to_tensor(results.x).flatten(), maxcg_iters)
Jacv_seventy = Jac_fn_lastdiff(numpy_to_tensor(results.x).flatten(), maxcg_iters, 0.7) @ v
Jacv_ninety = Jac_fn_lastdiff(numpy_to_tensor(results.x).flatten(), maxcg_iters, 0.9) @ v
Jacv_ninety_five = Jac_fn_lastdiff(numpy_to_tensor(results.x).flatten(), maxcg_iters, 0.95) @ v
Jacv_all = Jac_fn_lastdiff(numpy_to_tensor(results.x).flatten(), maxcg_iters, 1) @ v
Jacv_no = Jac_fn_lastdiff(numpy_to_tensor(results.x).flatten(), maxcg_iters, 0) @ v

for h in H:
    seventy.append(torch.norm(res_fn(numpy_to_tensor(results.x).flatten() + h * v, maxcg_iters) - f_x - h * Jacv_seventy).detach())
    ninety.append(torch.norm(res_fn(numpy_to_tensor(results.x).flatten() + h * v, maxcg_iters) - f_x - h * Jacv_ninety).detach())
    ninety_five.append(torch.norm(res_fn(numpy_to_tensor(results.x).flatten() + h * v, maxcg_iters) - f_x - h * Jacv_ninety_five).detach())
    all_grad.append(torch.norm(res_fn(numpy_to_tensor(results.x).flatten() + h * v, maxcg_iters) - f_x - h * Jacv_all).detach())
    no_grad.append(torch.norm(res_fn(numpy_to_tensor(results.x) + h * v, maxcg_iters) - f_x - h * Jacv_no).detach())

plt.plot(H, seventy, marker='.', markersize=10, label='70%')
plt.plot(H, ninety, marker='.', markersize=10, label='90%')
plt.plot(H, ninety_five, marker='.', markersize=10, label='95%')
plt.plot(H, all_grad, marker='.', markersize=10, label='100%')
plt.plot(H, no_grad, marker='.', markersize=10, label='0%')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlabel("Perturbation Multiplier")
plt.ylabel("Error")
plt.title(f"Differentiating Last % of Iters")

plt.savefig('results/figs/diff_CG_iters_deriv.pdf')

print(80*'-')
print('COMPLETED PLOTTING \'percCG\' TEST')
print(80*'-')






f0_grads = []
loss_grads = []
grad_grads = []

print(80*'-')
print('INITIATING \'percCG\' FIRSTITER TEST')
print(80*'-')

for prop in proportions:
    print(80*'-')
    print('INITIATION \'percCG\' TEST AT PROPORTION {}'.format(prop))
    print(80*'-')

    wp_list,_,losses,grads = lsq_lma(numpy_to_tensor(results.x).flatten(), 
                                     res_fn_firstdiff, 
                                     Jac_fn_firstdiff, 
                                     args=[maxcg_iters, prop], 
                                     gtol=0, 
                                     max_iter=max_iter, 
                                     verbose=False, 
                                     return_loss_and_grad=True)
    wp_vec = wp_list[-1]
    def A_forward(f):
        reference = f.reshape(*domain_R.m)        
        reference_img = SplineInter(reference, domain_R)
        template_predictions = torch.hstack([Forward_single(wp_vec.reshape(-1,6)[j], ys[j], reference_img)[0] for j in range(n_images)])
        regularizer = lam * L(f)
        return torch.hstack([template_predictions * torch.sqrt(torch.prod(domain_T.h)),
                                regularizer * torch.sqrt(torch.prod(domain_R.h))])
    A = LinearOperator((torch.prod(domain_R.m).item(),), forward_fun = A_forward, dtype = torch.float64)
    f0 = conjugate_gradient_nograd(A.T ^ A, A.T @ b, f0_interp, max_iter= 5000, tol=1e-12)
    f0_grad_reshaped = f0.reshape(*domain_R.m)

    f0_grads.append(f0_grad_reshaped.clone().detach())
    loss_grads.append(losses)
    grad_grads.append(grads)

    print(80*'-')
    print('COMPLETED \'percCG\' TEST AT PROPORTION {}'.format(prop))
    print(80*'-')

print(80*'-')
print('COMPLETED \'percCG\' FIRSTITER TEST')
print(80*'-')



print(80*'-')
print('PLOTTING \'percCG\' FIRSTITER TEST')
print(80*'-')


maxcg_iters = 500  # verified that all iterations reach max_iters before stopping
proportions = [0.7, 0.9, 0.95, 1]

H = torch.logspace(5, -15, 10)
# Loop through each subplot and plot

seventy = []
ninety = []
ninety_five = []
all_grad = []
no_grad = []
v = Jac_fn_lastdiff(numpy_to_tensor(results.x).flatten(), maxcg_iters, 1).T @ res_fn_nodiff(numpy_to_tensor(results.x).flatten(), maxcg_iters)
v /= torch.norm(v)
f_x = res_fn_nodiff(numpy_to_tensor(results.x).flatten(), maxcg_iters)
Jacv_seventy = Jac_fn_lastdiff(numpy_to_tensor(results.x).flatten(), maxcg_iters, 0.7) @ v
Jacv_ninety = Jac_fn_lastdiff(numpy_to_tensor(results.x).flatten(), maxcg_iters, 0.9) @ v
Jacv_ninety_five = Jac_fn_lastdiff(numpy_to_tensor(results.x).flatten(), maxcg_iters, 0.95) @ v
Jacv_all = Jac_fn_lastdiff(numpy_to_tensor(results.x).flatten(), maxcg_iters, 1) @ v
Jacv_no = Jac_fn_lastdiff(numpy_to_tensor(results.x).flatten(), maxcg_iters, 0) @ v

for h in H:
    seventy.append(torch.norm(res_fn(numpy_to_tensor(results.x).flatten() + h * v, maxcg_iters) - f_x - h * Jacv_seventy).detach())
    ninety.append(torch.norm(res_fn(numpy_to_tensor(results.x).flatten() + h * v, maxcg_iters) - f_x - h * Jacv_ninety).detach())
    ninety_five.append(torch.norm(res_fn(numpy_to_tensor(results.x).flatten() + h * v, maxcg_iters) - f_x - h * Jacv_ninety_five).detach())
    all_grad.append(torch.norm(res_fn(numpy_to_tensor(results.x).flatten() + h * v, maxcg_iters) - f_x - h * Jacv_all).detach())
    no_grad.append(torch.norm(res_fn(numpy_to_tensor(results.x) + h * v, maxcg_iters) - f_x - h * Jacv_no).detach())

plt.plot(H, seventy, marker='.', markersize=10, label='70%')
plt.plot(H, ninety, marker='.', markersize=10, label='90%')
plt.plot(H, ninety_five, marker='.', markersize=10, label='95%')
plt.plot(H, all_grad, marker='.', markersize=10, label='100%')
plt.plot(H, no_grad, marker='.', markersize=10, label='0%')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.xlabel("Perturbation Multiplier")
plt.ylabel("Error")
plt.title(f"Differentiating Last % of Iters")

plt.savefig('results/figs/diff_CG_iters_deriv.pdf')

print(80*'-')
print('COMPLETED \'percCG\' FIRSTITER TEST')
print(80*'-')
