import torch
from src.domain import Domain
from src.interpolation import SplineInter

# Some Math Operations that keep coming up in my work --Cash

def subtract_linear_trend(data, dim=2):
    # Get the size of the dimension along which to detrend
    n = data.shape[dim]

    # Create a tensor of indices along the dimension and normalize it
    indices = torch.arange(n, dtype=torch.float32, device=data.device).unsqueeze(0)
    indices = indices / (n - 1)  # Normalize to [0, 1]

    # Add a constant term (bias) to the model by concatenating ones
    X = torch.cat([indices, torch.ones_like(indices)], dim=0).T

    # Reshape data to flatten the other dimensions except for the detrending dimension
    original_shape = data.shape
    num_elements = data.numel() // n
    data_flat = data.transpose(dim, -1).reshape(num_elements, n)

    # Apply least squares regression to find coefficients
    # Solve (X^T X)^(-1) X^T y
    XT_X = X.T @ X
    XT_y = X.T @ data_flat.T
    beta = torch.linalg.solve(XT_X, XT_y)

    # Compute the trend for each element
    trend = X @ beta

    # Subtract the trend from the original data, reshaped back to the original structure
    detrended_data = (data_flat.T - trend).reshape(original_shape).transpose(dim, -1)

    return detrended_data

def finDiffInterp(data,domain,dim=0,order=1):
    diff = torch.diff(data,dim=dim,n=order) / ( (domain.omega[2 * dim + 1] - domain.omega[2 * dim]) / domain.m[dim] )
    newm = domain.m.clone()

    newm[dim] -= 1

    diff_domain = Domain(domain.omega,newm)

    return SplineInter(diff, diff_domain)


if  __name__ == '__main__':
    # Test subtract_linear_trend
    tensor = torch.randn(2,5,2) + torch.einsum('ik,j->ijk',torch.ones(2,2),torch.arange(5)) # Example tensor
    detrended_tensor = subtract_linear_trend(tensor, dim=1)
    print("Tensor with linear trend :" , tensor)
    print("Tensor without linear trend :", detrended_tensor)

    data = torch.outer(torch.ones(50),torch.linspace(0,5,50))
    domain = Domain([0.,4.,0.,4.],[50,50])
    data_y = finDiffInterp(data,domain,dim=1)
    print("data on grid [0.,4.,0.,4.]: ",data)
    xc = torch.outer(torch.linspace(0,5,11),torch.ones(2))
    print("interpolated data at x = {} : ".format(xc), data_y(xc))