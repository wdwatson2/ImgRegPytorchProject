import matplotlib.pyplot as plt
import numpy as np
from src.domain import Domain
import torch
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize

# Plotting functions need to be rewritten to move domain constructor variables to cpu in the future if we want cross compatibility on cpu and gpu

def view_image_2d(image, domain, kwargs={'cmap':'gray'}, ax = None):
    if ax == None:
        return plt.imshow(image.reshape(domain.m[0],domain.m[1]).T, origin='lower', extent=(domain.omega[0], domain.omega[1], domain.omega[2], domain.omega[3]), **kwargs)
    return ax.imshow(image.reshape(domain.m[0],domain.m[1]).T, origin='lower', extent=(domain.omega[0], domain.omega[1], domain.omega[2], domain.omega[3]), **kwargs)

def view_contour_2d(image, domain, kwargs={}):
    # Assuming domain.m and domain.omega are tensors, convert them to NumPy arrays
    m = domain.m.numpy() if isinstance(domain.m, torch.Tensor) else domain.m
    omega = domain.omega.numpy() if isinstance(domain.omega, torch.Tensor) else domain.omega

    # Reshape the image appropriately, ensuring it's a NumPy array for plotting
    reshaped_image = image.reshape(m[0], m[1]).T.numpy() if isinstance(image, torch.Tensor) else image.reshape(m[0], m[1]).T

    # Create the meshgrid and plot the contour
    return plt.contour(reshaped_image, origin='lower', extent=(omega[0], omega[1], omega[2], omega[3]), **kwargs)

def view_contourf_2d(image, domain, kwargs={}):
    # Assuming domain.m and domain.omega are tensors, convert them to NumPy arrays
    m = domain.m.numpy() if isinstance(domain.m, torch.Tensor) else domain.m
    omega = domain.omega.numpy() if isinstance(domain.omega, torch.Tensor) else domain.omega

    # Reshape the image appropriately, ensuring it's a NumPy array for plotting
    reshaped_image = image.reshape(m[0], m[1]).T.numpy() if isinstance(image, torch.Tensor) else image.reshape(m[0], m[1]).T

    # Create the meshgrid and plot the contour
    return plt.contour(reshaped_image, origin='lower', extent=(omega[0], omega[1], omega[2], omega[3]), **kwargs)


def plot_grid_2d(grid, domain, spacing=1, kwargs={}):
    grid = grid.reshape(domain.m[0]+1, domain.m[1]+1, 2)
    ax = plt.gca()
    mesh = ax.pcolormesh(grid[::spacing,::spacing,0], grid[::spacing,::spacing,1], np.ones((grid.shape[0]//spacing, grid.shape[1]//spacing)), edgecolors='w',alpha=0.1, antialiased=True, **kwargs)

    #plt.axis('equal')

# this function visualizes a 3d tensor using orthogonal slices in 2d. the slices are all combined in a 3d plot
def view_image_3d(image, domain, sx=None,sy=None,sz=None, kwargs={}):
    if sx is None:
        sx = [domain.m[0]//2]
    if sy is None:
        sy = [domain.m[1]//2]
    if sz is None:
        sz = [domain.m[2]//2]

    T = np.reshape(image.numpy(), tuple(domain.m.numpy()))
    norm = Normalize(vmin=T.min(), vmax=T.max())
    h = domain.h.numpy()

    ax = plt.gca()

    # show ortho-slices to first dimension
    x1 = domain.xc(0).numpy()
    x2 = domain.xc(1).numpy()
    x3 = domain.xc(2).numpy()

    for i in sx:
        Y,Z = np.meshgrid(x2,x3)
        X = (domain.omega[0] + (i - 0.5) * domain.h[0]) * np.ones(tuple(domain.m[1:]))
        Ti = np.squeeze(T[i, :, :])
        ax.plot_surface(X, Y.T, Z.T, facecolors=plt.cm.viridis(norm(Ti)))

    # show ortho-slices to second dimension
    for i in sy:
        Y = (domain.omega[2] + (i - 0.5) * domain.h[1]) * np.ones([domain.m[0].item(), domain.m[2].item()])
        X,Z = np.meshgrid(x1,x3)
        Ti = np.squeeze(T[:, i, :])
        ax.plot_surface(X.T,Y,Z.T, facecolors=plt.cm.viridis(norm(Ti)))

    # show ortho-slices to third dimension
    for i in sz:
        Z = (domain.omega[4] + (i - 0.5) * domain.h[2]) * np.ones(tuple(domain.m[:2]))
        X,Y = np.meshgrid(x1,x2)
        Ti = np.squeeze(T[:, :, i])
        ax.plot_surface(X.T, Y.T, Z, facecolors=plt.cm.viridis(norm(Ti)))
 

    





if  __name__ == '__main__':
    domain = Domain(torch.tensor([0.0, 2.0, 0.0,  1.0]), torch.tensor([5,8]))
    xc = domain.getCellCenteredGrid()
    fig, axs = plt.subplots(2, 3, figsize=(10, 5))
    plt.subplot(2,3, 1)
    view_image_2d(xc[:,:,0], domain)
    xn = domain.getNodalGrid()
    plot_grid_2d(xn, domain)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.subplot(2,3, 2)
    view_image_2d(xc[:,:,1], domain)
    plt.xlabel('x')
    plt.ylabel('y')
    
    domain_3d = Domain(torch.tensor([0.0, 2.0, 0.0,  1.0, 0.0, 1.0]), torch.tensor([4,8, 16]))  
    xc_3d = domain_3d.getCellCenteredGrid()
    ax = plt.subplot(2,3, 4, projection='3d')
    view_image_3d(xc_3d[:,:,:,0], domain_3d)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.clabel('z')
    ax = plt.subplot(2,3, 5, projection='3d')
    view_image_3d(xc_3d[:,:,:,1], domain_3d)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.clabel('z')
    ax = plt.subplot(2,3, 6, projection='3d')
    view_image_3d(xc_3d[:,:,:,2], domain_3d)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.clabel('z')
    
    plt.show()




