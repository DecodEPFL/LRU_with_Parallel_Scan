import torch
import math


class PScan(torch.autograd.Function):
    @staticmethod
    def pscan(A, X):
        # A : (B, D, L, N)
        # X : (B, D, L, N)

        # modifies X in place by doing a parallel scan.
        # more formally, X will be populated by these values :
        # H[t] = A[t] * H[t-1] + X[t] with H[0] = 0
        # which are computed in parallel (2*log2(T) sequential steps (ideally), instead of T sequential steps)

        B, D, L, _ = A.size()
        num_steps = int(math.log2(L))

        # up sweep or reduction step
        Aa = A
        Xa = X
        for k in range(num_steps):
            T = 2 * (Xa.size(2) // 2)

            Aa = Aa[:, :, :T].view(B, D, T // 2, 2, -1)
            Xa = Xa[:, :, :T].view(B, D, T // 2, 2, -1)

            Xa[:, :, :, 1].add_(Aa[:, :, :, 1].mul(Xa[:, :, :, 0]))
            Aa[:, :, :, 1].mul_(Aa[:, :, :, 0])

            Aa = Aa[:, :, :, 1]
            Xa = Xa[:, :, :, 1]

        # down sweep
        for k in range(num_steps - 1, -1, -1):
            Aa = A[:, :, 2**k - 1 : L : 2**k]
            Xa = X[:, :, 2**k - 1 : L : 2**k]

            T = 2 * (Xa.size(2) // 2)

            if T < Xa.size(2):
                Xa[:, :, -1].add_(Aa[:, :, -1].mul(Xa[:, :, -2]))
                Aa[:, :, -1].mul_(Aa[:, :, -2])

            Aa = Aa[:, :, :T].view(B, D, T // 2, 2, -1)
            Xa = Xa[:, :, :T].view(B, D, T // 2, 2, -1)

            Xa[:, :, 1:, 0].add_(
                Aa[:, :, 1:, 0].mul(Xa[:, :, :-1, 1])
            )
            Aa[:, :, 1:, 0].mul_(Aa[:, :, :-1, 1])

    @staticmethod
    def forward(ctx, A_in, X_in):
        """
        Applies the parallel scan operation, as defined above. Returns a new tensor.

        Args:
            A_in : (B, L, D, N)
            X_in : (B, L, D, N)

        Returns:
            H : (B, L, D, N)
        """

        # clone tensor (in-place ops)
        A = A_in.clone()  # (B, L, D, N)
        X = X_in.clone()  # (B, L, D, N)

        # prepare tensors
        A = A.transpose(2, 1)  # (B, D, L, N)
        X = X.transpose(2, 1)  # (B, D, L, N)

        # parallel scan
        PScan.pscan(A, X)

        ctx.save_for_backward(A_in, X)

        return X.transpose(2, 1)

    @staticmethod
    def backward(ctx, grad_output_in):
        """
        Flows the gradient from the output to the input. Returns two new tensors.

        Args:
            ctx : A_in : (B, L, D, N), X : (B, D, L, N)
            grad_output_in : (B, L, D, N)

        Returns:
            gradA : (B, L, D, N), gradX : (B, L, D, N)
        """

        A_in, X = ctx.saved_tensors

        # clone tensors
        A = A_in.clone()
        # grad_output_in will be cloned with flip()

        # prepare tensors
        A = A.transpose(2, 1)  # (B, D, L, N)
        A = torch.cat((A[:, :, :1], A[:, :, 1:].flip(2)), dim=2)
        grad_output_b = grad_output_in.transpose(2, 1)

        # reverse parallel scan
        grad_output_b = grad_output_b.flip(2)
        PScan.pscan(A, grad_output_b)
        grad_output_b = grad_output_b.flip(2)

        Q = torch.zeros_like(X)
        Q[:, :, 1:].add_(X[:, :, :-1] * grad_output_b[:, :, 1:])

        return Q.transpose(2, 1), grad_output_b.transpose(2, 1)


pscan = PScan.apply


T=10
A=torch.randn(2,2)
A=A.unsqueeze(0)
A=torch.tile(A,(T,1,1))
A=A.unsqueeze(0)


B=torch.randn(2,2)
B=B.unsqueeze(0)
B=torch.tile(B,(T,1,1))


U=torch.randn((10,2,1))

X=torch.matmul(B,U)

padded_result = torch.zeros((T, 2,2))

padded_result[:, :, :1] = X

X=padded_result

X=X.unsqueeze(0)

Y=pscan(A, X)

Y

Yv=torch.sum(Y, dim=2)

X=X.squeeze()
Y=Y.squeeze()
A=A.squeeze()

x0=X[0,:,:]
a0=A[0,:,:]
a1=A[1,:,:]
y0=Y[0,:,:]
y1=Y[1,:,:]
x1=X[1,:,:]




a2=A[2,:,:]
x2=X[2,:,:]
y2=Y[2,:,:]


y11=a0*y0+x1

Yt=torch.zeros((10,2))
Yt[0,:]= torch.sum(X[0,:,:], dim=1)

for k in range(T-1):
    Yt[k+1,:] = A[k,:,:]@Yt[k,:]+torch.sum(X[k+1,:,:], dim=1)

    # Compute B using the Kronecker product of each A[k,:,:] with the identity matrix I
    I_n = torch.eye(2)  # Identity matrix of shape (n, n)
    B = torch.kron(A, I_n).reshape(T, 2 * 2, 2 * 2)  # Shape (T, n*n, n*n)

    # Flatten X along the last two dimensions to match the flattened structure of Y
    H = X.reshape(T, 2 * 2)













Yt


