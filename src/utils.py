import torch

# write the parameters of trafo into a vector
def params_to_vec(trafo):
    return torch.cat([p.view(-1) for p in trafo.parameters()])

# unpack the parameters of trafo from a vector
def vec_to_params(trafo, params):
    start = 0
    for p in trafo.parameters():
        end = start + p.numel()
        p.data = params[start:end].view_as(p)
        start = end

# given a 1D vector of parameters and a dictionary with the keys and their shapes
import numpy as np
def vec_to_paramdic(vec, dic):
    idx = 0
    output = {}
    for k, shape in dic.items():
        nextidx = idx+np.prod(shape)
        output[k] = vec[idx:nextidx].reshape(shape)
        idx = nextidx
    return output

# count the number of parameters of trafo
def count_params(trafo):
    return sum(p.numel() for p in trafo.parameters())

def flatten_params(param_dict):
    params, shapes, sizes = [], [], []
    for p in param_dict.values():
        params.append(p.view(-1))
        shapes.append(p.shape)
        sizes.append(p.numel())
    flat_params = torch.cat(params)
    return flat_params, shapes, sizes

def unflatten_params(flat_params, keys, shapes, sizes):
    params_dict = {}
    i = 0
    for name, shape, size in zip(keys, shapes, sizes):
        params_dict[name] = flat_params[i:i+size].view(shape)
        i += size
    return params_dict

def flatten_params_list(param_dict_list):
    flat_params_list = []
    shapes_list = []
    sizes_list = []
    for param_dict in param_dict_list:
        flat_params, shapes, sizes = flatten_params(param_dict)

        flat_params_list.append(flat_params)
        shapes_list.append(shapes)
        sizes_list.append(sizes)
    return flat_params_list, shapes_list, sizes_list 

def unflatten_params_list(flat_params_list, keys_list, shapes_list, sizes_list):
    return [unflatten_params(flat_params, keys, shapes, sizes) for flat_params, keys, shapes, sizes in zip(flat_params_list, keys_list, shapes_list, sizes_list)]

def extract_jac_data(batch_jac, nbatch, nclasses):
    """
    Extract data stored in dict and store as 3D array
    """
    theta = None
    for keys, jacs in batch_jac.items():
        if theta is None:
            theta = jacs.reshape(nbatch, nclasses, -1)
        else:
            theta = torch.cat((theta, jacs.reshape(nbatch, nclasses, -1)), dim=2)
        batch_jac[keys] = None
        torch.cuda.empty_cache()
        
    return theta

def extract_jac_data_SR(jac_dict, T_pred_shape):
    output = None
    for i in range(T_pred_shape[0]):
        theta = None
        for keys, jacs in jac_dict.items():
            if theta is None:
                theta = jacs[i].reshape(T_pred_shape[1], T_pred_shape[2], -1)
            else:
                theta = torch.cat((theta, jacs.reshape(T_pred_shape[1], T_pred_shape[2], -1)), dim=2)
            jac_dict[keys] = None
            torch.cuda.empty_cache()
        if output == None:
            output = theta
        else:
            output = torch.cat(theta, dim=2)
    return output
    

def numpy_to_tensor(numpy_array):
    return torch.tensor(numpy_array, dtype=torch.float64, requires_grad=True)

def tensor_to_numpy(tensor):
    return tensor.detach().numpy()