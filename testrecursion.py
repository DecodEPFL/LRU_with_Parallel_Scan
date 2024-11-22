import torch

# Define A, u_k, and initial x_0
A = torch.tensor([[2.0, 1.0], [1.0, 3.0]])  # A matrix (2x2)
u_k = torch.tensor([1.0, 0.0])  # Control vector u_k (2,)
x_0 = torch.tensor([1.0, 2.0])  # Initial state x_0 (2,)


# Recursion using matrix multiplication
def recursion_matrix_mult(A, x_0, u_k, T):
    x = torch.zeros((T, 2), dtype=A.dtype)
    x[0] = x_0
    for k in range(1, T):
        x[k] = A @ x[k - 1] + u_k
    return x


# Recursion using element-wise multiplication
def recursion_elementwise_mult(A, u_k, x_0, T):
    # Initialize Y_k with the same shape as x_k
    Y = torch.zeros((T, 2), dtype=A.dtype)
    Y[0] = x_0

    # Element-wise recurrence (Hadamard product and addition)
    for k in range(1, T):
        B = A
        H_k = u_k
        Y[k] = B[0, 0] * Y[k - 1] + H_k
    return Y


# Run both recursions
T = 3  # Number of steps
x_matrix_mult = recursion_matrix_mult(A, x_0, u_k, T)
x_elementwise_mult = recursion_elementwise_mult(A, u_k, x_0, T)

print("Matrix multiplication recursion:")
print(x_matrix_mult)

print("Element-wise multiplication recursion:")
print(x_elementwise_mult)

Y