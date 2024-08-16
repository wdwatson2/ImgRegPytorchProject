import torch

def conjugate_gradient(A, b, x0, tol=1e-9, grad=True, max_iters=None, grad_iters=0):
    """ https://indrag49.github.io/Numerical-Optimization/conjugate-gradient-methods-1.html 
    solve Ax = b for x

    Input:
        A - a symmetric positive-definite n x n LinearOperator or Tensor
        b - a 1D pytorch tensor of length n
        x0 - a 1D pytorch tensor of length n which is the initial guess
        tol - an optional parameter, will stop when norm of residual < tol
        extension - number of iterations to use 

    best to use torch.float64 for all dtypes
    
    """
    with torch.no_grad():
        if max_iters == None:
            max_iters = x0.numel()
        x = x0.flatten() # x is R_hat
        r = b - A(x) # residual
        if torch.norm(r) < tol: return x
        delta = r # conjugate-gradient direction

        for _ in range(max_iters - 1):
            A_delta = A(delta)
            # D.adjoint_func = None
            # D.output_shape = None
            beta = (r @ r) / (delta @ A_delta)
            x = x + beta * delta
            r_new = r - beta * A_delta
            if torch.norm(r_new) < tol:
                # print('found minimizer in ' + str(i+1) + ' iterations')
                break
            chi = (r_new @ r_new) / (r @ r)
            delta = chi * delta + r_new

            r = r_new
    
    for _ in range(grad_iters):
        A_delta = A(delta)
        # D.adjoint_func = None
        # D.output_shape = None
        beta = (r @ r) / (delta @ A_delta)
        x = x + beta * delta
        r_new = r - beta * A_delta
        chi = (r_new @ r_new) / (r @ r)
        delta = chi * delta + r_new

        r = r_new
    # print('reached max_iters iterations to find minimizer')
    return x
        

    
def play():

    a = torch.rand(10)
    b = torch.rand(10)

    print(a @ b)

def identity_test():
    # stops after 1 iteration
    A = torch.eye(10, dtype=torch.float64)
    A_func = lambda x: A @ x

    b = torch.rand(10, dtype=torch.float64)

    x = conjugate_gradient(A_func, b, torch.zeros(10, dtype=torch.float64))
    print('x - b', x - b)

def random_test():
    n = 10

    a = torch.rand((n, n), dtype=torch.float64)-0.5
    A = a @ a.T
    print('condition number of A: ', torch.linalg.cond(A))
    A_func = lambda x: A @ x

    b = torch.rand(n, dtype=torch.float64)

    x = conjugate_gradient(A_func, b, torch.zeros(n, dtype=torch.float64))
    print('A @ x - b', A @ x - b)

def negative_test():
    A = -1 * torch.eye(10, dtype=torch.float64)
    A_func = lambda x: A @ x

    b = torch.rand(10, dtype=torch.float64)
    x0 = torch.rand(10, dtype=torch.float64)

    x = conjugate_gradient(A_func, b, x0)
    print('x', x)
    print('b', b)
    print('A @ x - b', A @ x - b)

def lanczos_test():
    from lanczos import lanczos_tridiag

    n = 100

    a = torch.rand((n, n))*2-0.5
    A = a @ a.T
    A_func = lambda x: A @ x

    b = torch.rand(n)

    x_c = conjugate_gradient(A_func, b, torch.rand(n), tol=1e-10)
    lT, lV = lanczos_tridiag(A_func, b, n, 1e-10, doReorth=True)
    x_l = (lV @ torch.linalg.solve(lT, lV.T @ b))

    print('conjugate gradient Ax-b: ', A @ x_c - b)
    print('lanczos Ax-b', A @ x_l - b)


if __name__ == '__main__':
    random_test()

# DONE
# solve problem with A = identity
# A is random 10 x 10, 10 steps should be right
# float 64
# check if beta becomes negative, somehow works and I didn't even address it!
# can test if A is not symmetric positive definite, should stop once encounters negative beta # with that too!
# I must be doing something wrong if the residual is going up and then suddenly down poof perfect
# compare with lanczos on simple matrix
# stopping criteria is relative residual below threshhold