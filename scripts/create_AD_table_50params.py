
import timeit
import matplotlib.pyplot as plt
import statistics
import pandas as pd
from src.transformations import VelocityNN, NeuralODE

SETUP_CODE = '''
# load hands-R.jpg and convert to pytorch tensor
import torch
from PIL import Image
import matplotlib.pyplot as plt
from src.domain import Domain

from src.utils import flatten_params, unflatten_params, vec_to_params, params_to_vec, count_params
import numpy as np
import torch.func as func
from src.transformations import VelocityNN, NeuralODE

torch.set_default_dtype(torch.float64)

R = Image.open('data/hands-R.jpg')
m = 32
R = R.resize((m, m))
R = torch.fliplr(torch.tensor(R.getdata(), dtype=torch.float32).view(m,m).transpose(0,1))

T = Image.open('data/hands-T.jpg')
T = T.resize((m, m))
T = torch.fliplr(torch.flipud(torch.tensor(T.getdata(), dtype=torch.float32).view(m,m).transpose(0,1)))

domain = Domain(torch.tensor([0.0, 20.0 ,0.0, 25.0]), torch.tensor([m, m]))
theta = torch.tensor(1e-3) 

from src.distance import SSDDistance
distance = SSDDistance(domain)

from src.interpolation import SplineInter

vel = VelocityNN(domain.dim,[domain.dim+1, 8]) 
trafo = NeuralODE(vel)
num_params = count_params(trafo)

keys = [k for k, _ in trafo.named_parameters()]
_, shapes, sizes = flatten_params({k: v.detach() for k, v in trafo.named_parameters()})
xc = domain.getCellCenteredGrid().view(-1,2)

def forward(wc, theta):
    wp = unflatten_params(wc,keys,shapes,sizes)
    yc = func.functional_call(trafo,wp,xc)
    Timg = SplineInter(T, domain,regularizer='moments',theta=theta)
    Ty = Timg(yc)
    return Ty

def lossfn(wc, theta):
    Ty = forward(wc, theta)
    Rimg = SplineInter(R, domain,regularizer='moments',theta=theta)
    Rc = Rimg(xc)
    loss = distance(Ty,Rc)
    return loss
'''

repeat = 2000
number = 1

def loss_timing():
    TEST_CODE = '''
wc = torch.randn(num_params)
lossfn(wc, theta)'''
    
    times_loss = timeit.repeat(setup=SETUP_CODE,
                            stmt=TEST_CODE,
                            repeat=repeat,
                            number=number)
    
    return times_loss

def grad_timing():
    TEST_CODE = '''
wc = torch.randn(num_params)
func.grad(lossfn, argnums=0)(wc, theta)'''
    
    times_grad = timeit.repeat(setup=SETUP_CODE,
                            stmt=TEST_CODE,
                            repeat=repeat,
                            number=number)
    return times_grad
    
def hess_timing():
    '''
    using func.jacrev(func.grad) is quicker than func.hessian for this problem
    '''

    TEST_CODE = '''
wc = torch.randn(num_params)
func.jacrev(func.grad(lossfn))(wc, theta)
'''
    
    times_hessian = timeit.repeat(setup=SETUP_CODE,
                            stmt=TEST_CODE,
                            repeat=repeat,
                            number=number)
    return times_hessian

def jacfwd_timing():
    TEST_CODE = '''
wc = torch.randn(num_params)
torch.func.jacfwd(forward, argnums=0)(wc, theta)'''
    
    times_jacfwd = timeit.repeat(setup=SETUP_CODE,
                            stmt=TEST_CODE,
                            repeat=repeat,
                            number=number)
    return times_jacfwd

def jacrev_timing():
    TEST_CODE = '''
wc = torch.randn(num_params)
torch.func.jacrev(forward, argnums=0)(wc, theta)'''
    
    times_jacrev = timeit.repeat(setup=SETUP_CODE,
                            stmt=TEST_CODE,
                            repeat=repeat,
                            number=number)
    return times_jacrev

def forward_timing():
    TEST_CODE = '''
wc = torch.randn(num_params)
forward(wc, theta)'''
    
    times_forward = timeit.repeat(setup=SETUP_CODE,
                            stmt=TEST_CODE,
                            repeat=repeat,
                            number=number)
    return times_forward  

def compute_stats(times):
    mean = statistics.mean(times)
    std = statistics.stdev(times)
    return mean, std


if __name__ == '__main__':
    
    
    times_loss = loss_timing()
    print("loss done")
    times_grad = grad_timing()
    print("grad done")
    times_hessian = hess_timing()
    print("hess done")
    times_forward = forward_timing()
    print("fwd done")
    times_jacfwd = jacfwd_timing()
    print("jacfwd done")
    times_jacrev = jacrev_timing()
    print("jacrev done")

    data = {}

    # Calculating statistics for each function
    data['loss'] = compute_stats(times_loss)
    data['grad'] = compute_stats(times_grad)
    data['hessian'] = compute_stats(times_hessian)
    data['forward'] = compute_stats(times_forward)
    data['jacfwd'] = compute_stats(times_jacfwd)
    data['jacrev'] = compute_stats(times_jacrev)

    # Creating a DataFrame
    df = pd.DataFrame(data, index=['mean', 'std']).transpose()

    print(df)   
    
    plt.figure()
    plt.subplot(2,3,1)
    plt.hist(times_loss)
    plt.xlabel('exec time')
    plt.title('Loss Eval')

    plt.subplot(2,3,2)
    plt.hist(times_grad)
    plt.xlabel('exec time')
    plt.title('Grad Eval')

    plt.subplot(2,3,3)
    plt.hist(times_hessian)
    plt.xlabel('exec time')
    plt.title('Hess Eval')

    plt.subplot(2,3,4)
    plt.hist(times_forward)
    plt.xlabel('exec time')
    plt.title('Forward Eval')

    plt.subplot(2,3,5)
    plt.hist(times_jacfwd)
    plt.xlabel('exec time')
    plt.title('JacFwd Eval')

    # plt.subplot(2,3,6)
    # plt.hist(times_jacrev)
    # plt.xlabel('exec time')
    # plt.title('JacRev Eval')
    
    plt.show()

    