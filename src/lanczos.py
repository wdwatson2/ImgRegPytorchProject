# this file contains Lanczos algorithms in pytorch
import torch


def lanczos_tridiag(A, b, k, tol, doReorth=False):
    """
    Lanczos method for computing the factorization

    A = Vk*Tk*Vk',

    where A is a real symmetric n by n matrix, Tk is a tridiagonal k by k 
    matrix and the columns of the n by k matrix Vk are orthogonal.

    Implementation follows:
        Paige, C. C. (1972). 
        Computational variants of the Lanczos method for the eigenproblem. 
        IMA Journal of Applied Mathematics. 

    Input:
            A - function computing A*x, e.g., x -> A*x
            b - right hand side vector
            k - dimension of Krylov subspace
          tol - stopping tolerance
     doReorth - (default=false) set to true to perform full reorthogonalization

    Output:
            T - sparse tridiagonal matrix
            V - basis vectors
    """

    n = len(b)

    # pre-allocate space for tridiagonalization and basis
    beta = torch.zeros(k)
    alpha = torch.zeros(k)
    V = torch.zeros(n, k)

    beta[0] = torch.norm(b)
    V[:, 0] = b / beta[0]
    u = A(V[:, 0])

    for j in range(k - 1):
        alpha[j] = torch.dot(V[:, j], u)
        u = u - alpha[j] * V[:, j]
        if doReorth:  # full re-orthogonalization
            for i in range(j + 1):
                u = u - V[:, i] * torch.dot(V[:, i], u)
        gamma = torch.norm(u)
        V[:, j + 1] = u / gamma
        beta[j + 1] = gamma
        if beta[j + 1] < tol:
            break
        u = A(V[:, j + 1]) - beta[j + 1] * V[:, j]

    T = torch.diag(beta[1:j+1], -1) + torch.diag(alpha[0:j+1], 0) + torch.diag(beta[1:j+1], 1)
    V = V[:, 0:j+1]

    return T, V


if __name__ == "__main__":
    # Example: compute the Lanczos factorization of a random matrix
    # set default dtype to float64
    torch.set_default_dtype(torch.float64)
    n = 10
    A = torch.randn(n, n)
    A = A + A.t() + n * torch.eye(n)
    b = torch.randn(n)
    k = 11
    tol = 1e-20
    T, V = lanczos_tridiag(lambda x: A @ x, b, k, tol,True)
    print(T.shape, V.shape  )

    # check the factorization
    A_ = V @ T @ V.t()
    print("Error in factorization: ", torch.norm(A - A_)/torch.norm(A))
    print("Error in orthogonality: ", torch.norm(V.t() @ V - torch.eye(V.shape[1])))
    print("Error in symmetry: ", torch.norm(A_ - A_.t()))
    print("Error in tridiagonal: ", torch.norm(T - T.t()))