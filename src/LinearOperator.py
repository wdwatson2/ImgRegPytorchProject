# pytorch module for linear operators
import torch

class LinearOperator(torch.nn.Module):
    """
    A linear operator class that can be used to define custom linear operators
    """
    def __init__(self, input_shape, forward_fun, adjoint_fun=None, output_shape=None, dtype=torch.float32, device='cpu'):
        super(LinearOperator, self).__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.forward_fun = forward_fun
        self.adjoint_fun = adjoint_fun
        self.dtype = dtype
        self.device = device

    @classmethod
    def Identity(cls, input_shape, dtype=torch.float32, device='cpu'):
        return LinearOperator(input_shape, forward_fun= lambda x : x.clone(), dtype=dtype, device=device)


    def forward(self, x):
        """
        Forward pass of the linear operator
        """
        if x.ndim == len(self.input_shape):
            return self.forward_fun(x.view(*self.input_shape))
        elif x.ndim == len(self.input_shape) + 1:
            x = x.view(-1, *self.input_shape)
            return torch.vmap(self.forward_fun)(x)
        else:
            raise ValueError(f"Invalid input shape {x.shape}, expected {self.input_shape} or {self.input_shape + (1,)} ")

    def _get_adjoint(self):
        """
        Get the adjoint of the adjoint operator
        """
        y = torch.zeros(*self.input_shape, dtype=self.dtype, device=self.device)
        (out,vjpfun) = torch.func.vjp(self.forward_fun, y)
        def adj_fun(x):
            return vjpfun(x)[0]
        return adj_fun, out.shape
       

    def adjoint(self, x):
        """
        Adjoint pass of the linear operator
        """
        if self.adjoint_fun is not None:
            if x.ndim == len(self.input_shape):
                return self.adjoint_fun(x)
            elif x.ndim == len(self.input_shape) + 1:
                return torch.vmap(self.adjoint_fun)(x)
            else:
                raise ValueError(f"Invalid input shape {x.shape}, expected {self.input_shape} or {self.input_shape + (1,)} ")
        else:
            # If adjoint is not defined, use the forward pass and automatic differentation to compute the adjoint
            self.adjoint_fun, self.output_shape = self._get_adjoint()
            return self.adjoint(x) # try again
            
    
    def transpose(self):
        if self.adjoint_fun is not None and self.output_shape is not None:
            return LinearOperator(self.output_shape, self.adjoint_fun, self.forward_fun, self.input_shape, self.dtype, self.device)        
        else:
            self.adjoint_fun, self.output_shape = self._get_adjoint()
            return self.transpose()
        
    def compose(self, other):
        "Composes LinearOperators"
        if self.input_shape != other.output_shape:
            raise ValueError("Output shape of the first operator must match input shape of the second operator.")

        def new_forward_fun(x):
            return self.forward_fun(other.forward_fun(x))

        return LinearOperator(other.input_shape, new_forward_fun, output_shape=self.output_shape, dtype=self.dtype, device=self.device)
    
    def add(self, other):
        """
        Define the addition of two linear operators. The addition operation is applied element-wise on the results of both operators.

        Args:
            other (LinearOperator): The operator to add to this operator.

        Returns:
            LinearOperator: A new linear operator representing the addition of this operator with the other operator.

        Raises:
            ValueError: If the output shapes of both operators do not match.
        """
        if self.output_shape != other.output_shape:
            raise ValueError("Output shapes of both operators must match to perform addition.")

        def new_forward_fun(x):
            return self.forward_fun(x) + other.forward_fun(x)

        return LinearOperator(self.input_shape, new_forward_fun, output_shape=self.output_shape, dtype=self.dtype, device=self.device)

    @property
    def T(self):
        return self.transpose()

    def matmul(self, x):
        """
        Matrix multiplication
        """
        return self.forward(x)

    def rmatmul(self, x):
        """
        Right matrix multiplication
        """
        return self.adjoint(x)

    def __matmul__(self, x):
        return self.forward(x)

    def __rmatmul__(self, x):
        return self.adjoint(x)

    def __mul__(self, x):
        return self.forward(x)

    def __rmul__(self, x):
        return self.adjoint(x)

    def __call__(self, x):
        return self.forward(x)
    
    def __add__(self, other):
        """
        Overload the addition operator to add two linear operators.

        Args:
            other (LinearOperator): The operator to add to this one.

        Returns:
            LinearOperator: The result of adding the two operators.
        """
        return self.add(other)
    
    def __and__(self, scalar):
        """
        Overload the '&' operator to perform scalar multiplication. This is unconventional but used here for custom behavior.

        Args:
            scalar (int or float): The scalar to multiply the operator's output.

        Returns:
            LinearOperator: A new linear operator where the forward function output is scaled by the scalar.

        """
        def scaled_forward_fun(x):
            return self.forward_fun(x) * scalar
        return LinearOperator(self.input_shape, scaled_forward_fun, output_shape=self.output_shape, dtype=self.dtype, device=self.device)
    
    def __xor__(self, x):
        """
        Overload the xor operator to compose two linear operators.

        Args:
            other (LinearOperator): The operator to compose with.

        Returns:
            LinearOperator: The result of composing the two operators.
        """
        return self.compose(x)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.input_shape})"
    

if __name__ == "__main__":
    A = torch.randn(5,8)
    def forward_fun(x):
        return A @ x
    x = torch.randn(8,13)
    Aop = LinearOperator(x.shape, forward_fun)
    print(Aop)

    Ax = Aop @ x
    w = torch.randn_like(Ax)
    t1 = torch.dot(Ax.view(-1), w.view(-1))
    Atw = Aop.adjoint(w)
    t2 = torch.dot(Atw.view(-1), x.view(-1))
    print(f"t1 = {t1:1.4e}, t2 = {t2:1.4e}, rel_err = {(t1-t2)/t1:1.4e}")
    
    print("Testing batch mode")
    x = torch.randn(15,8,13)
    Ax = Aop(x)
    w = torch.randn_like(Ax)
    t1 = torch.sum(Ax * w)
    Atw = Aop.adjoint(w)
    t2 = torch.sum(Atw * x)
    print(f"t1 = {t1:1.4e}, t2 = {t2:1.4e}, rel_err = {(t1-t2)/t1:1.4e}")

    # testing derivatives
    print("Testing derivatives")
    A.requires_grad = True
    print(A.grad)
    Aop = LinearOperator(x.shape, forward_fun)
    loss = torch.sum(Aop(x) )
    # grad_A (e'*A*x)*e = e'*x*e
    loss.backward()
    print(A.grad)
    print(torch.sum(torch.sum(x, dim=0),-1))

    # testing adjoint of adjoint
    print("Testing adjoint of adjoint")
    Aop = LinearOperator(x.shape, forward_fun)
    Aop_adjoint = Aop.T
    Aop_adjoint_adjoint = Aop_adjoint.T
    # print(Aop_adjoint_adjoint(x)- Aop(x))
    print(torch.norm(Aop_adjoint_adjoint(x)- Aop(x)))

    # testing identity
    print("Testing Identity")
    Id = LinearOperator.Identity(x.shape)
    print(torch.norm(Id(x) - x))
