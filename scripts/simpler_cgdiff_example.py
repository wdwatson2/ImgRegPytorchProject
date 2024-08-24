import torch.func as func
import matplotlib.pyplot as plt
import torch

# a simple parametric matrix 
def A(theta, C):
    return theta[0] * torch.eye(C.size(0), dtype=torch.float64) + theta[1] * C

def conjugate_gradient(A_func, b, theta, x0, max_iter=1000, tol=1e-10):
    x = x0
    r = b - A_func(theta).matmul(x)
    p = r.clone()
    rsold = r.dot(r)

    for i in range(max_iter):
        Ap = A_func(theta).matmul(p)
        alpha = rsold / p.dot(Ap)
        x += alpha * p
        r -= alpha * Ap
        rsnew = r.dot(r)
        if torch.sqrt(rsnew) < tol:
            # print(f"stopping at {i}")
            break
        p = r + (rsnew / rsold) * p
        rsold = rsnew

    # print("max iter reached")
    return x

def conjugate_gradient_nograd(A_func, b, theta, x0, max_iter=1000, tol=1e-10):
    with torch.no_grad():
        x = x0
        r = b - A_func(theta).matmul(x)
        p = r.clone()
        rsold = r.dot(r)

        for i in range(max_iter):
            Ap = A_func(theta).matmul(p)
            alpha = rsold / p.dot(Ap)
            x += alpha * p
            r -= alpha * Ap
            rsnew = r.dot(r)
            if torch.sqrt(rsnew) < tol:
                # print(f"stopping at {i}")
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew

        # print("max iter reached")
        return x
    
def test_conjugate_gradient(seed):
    torch.manual_seed(seed)

    n = 50 # matrix dim
    theta = torch.tensor([1.0, 0.5], requires_grad=True, dtype=torch.float64) # 2 parameters
    C = torch.diag(torch.arange(1, n + 1, dtype=torch.float64))
    b = torch.randn(n, dtype=torch.float64)
    x0 = torch.zeros(n, dtype=torch.float64)

    def res(theta, C, b, x0):
        x = conjugate_gradient(lambda theta: A(theta, C), b, theta, x0)
        return (A(theta, C) @ x) - b
    
    def res_nograd(theta, C, b, x0):
        x = conjugate_gradient_nograd(lambda theta: A(theta, C), b, theta, x0)
        return (A(theta, C) @ x) - b
    
    def jacfn(theta, C, b, x0):
        return func.jacfwd(res)(theta, C, b, x0).detach()

    def jacfn_nograd(theta, C, b, x0):
        return func.jacfwd(res_nograd)(theta, C, b, x0).detach()

    H = torch.logspace(20,-20,10)

    no_diffs = []
    diffs = []
    const = []

    v = jacfn(theta, C, b, x0).T @ res(theta, C, b, x0) # in the direction of the gradient
    v /= torch.norm(v)

    res_normal = res(theta, C, b, x0)
    res_nodiff = res_nograd(theta, C, b, x0)
    Jacv_nodiff = jacfn_nograd(theta, C, b, x0) @ v
    Jacv = jacfn(theta, C, b, x0) @ v

    for h in H:
        no_diffs.append(torch.norm(res_nograd(theta + h * v, C, b, x0) - res_nodiff - h * Jacv_nodiff).detach())
        diffs.append(torch.norm(res(theta + h * v, C, b, x0) - res_normal - h * Jacv).detach())
        const.append(torch.norm(res(theta + h * v, C, b, x0) - res_normal).detach())

    plt.scatter(H, no_diffs, marker='x', label = 'No Differentiation through CG')
    plt.scatter(H, diffs, marker='+', label = 'Differentiation through all CG Iterations')
    plt.scatter(H, const, marker='o', label = 'Constant')

    plt.gca().set_xscale('log')
    plt.gca().set_yscale('log')

    plt.legend()

    plt.xlabel("Perturbation Multiplier")
    plt.ylabel("Error")

    plt.title("Derivative Check for Jacobian in Direction of Gradient")
    plt.show()

    print(torch.norm(jacfn(theta, C, b, x0) - jacfn_nograd(theta, C, b, x0)))

if __name__ == '__main__':

    test_conjugate_gradient(20)