import math
import torch
import torch.nn as nn
from sympy import false


class MLP(nn.Module): # Simple MLP layer used in the SSM scaffolding later on, can be modified
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        # Define the model using nn.Sequential
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=false),  # First layer
            nn.SiLU(),  # Activation after the first layer
            nn.Linear(hidden_size, hidden_size, bias=false),  # Hidden layer
            nn.ReLU(),  # Activation after hidden layer
            nn.Linear(hidden_size, output_size, bias=false)  # Output layer (no activation)
        )

    def forward(self, x):
        if x.dim() == 3:
            # x is of shape (batch_size, sequence_length, input_size)
            batch_size, seq_length, input_size = x.size()

            # Flatten the batch and sequence dimensions for the MLP
            x = x.reshape(-1, input_size)  # Use reshape instead of view

            # Apply the MLP to each feature vector
            x = self.model(x)  # Shape: (batch_size * sequence_length, output_size)

            # Reshape back to (batch_size, sequence_length, output_size)
            output_size = x.size(-1)
            x = x.reshape(batch_size, seq_length, output_size)  # Use reshape instead of view
        else:
            # If x is not 3D, just apply the MLP directly
            x = self.model(x)
        return x


class PScan(torch.autograd.Function): # Parallel Scan Algorithm
    # Given A is NxTx1 and X is NxTxD, expands A and X in place in O(T),
    # and O(log(T)) if not core-bounded, so that
    #
    # Y[:, 0] = Y_init
    # Y[:, t] = A[:, t] * Y[:, t-1] + X[:, t]
    #
    # can be computed as
    #
    # Y[:, t] = A[:, t] * Y_init + X[:, t]

    @staticmethod
    def expand_(A, X):
        if A.size(1) == 1:
            return
        T = 2 * (A.size(1) // 2)
        Aa = A[:, :T].view(A.size(0), T // 2, 2, -1)
        Xa = X[:, :T].view(X.size(0), T // 2, 2, -1)
        Xa[:, :, 1].add_(Aa[:, :, 1].mul(Xa[:, :, 0]))
        Aa[:, :, 1].mul_(Aa[:, :, 0])
        PScan.expand_(Aa[:, :, 1], Xa[:, :, 1])
        Xa[:, 1:, 0].add_(Aa[:, 1:, 0].mul(Xa[:, :-1, 1]))
        Aa[:, 1:, 0].mul_(Aa[:, :-1, 1])
        if T < A.size(1):
            X[:, -1].add_(A[:, -1].mul(X[:, -2]))
            A[:, -1].mul_(A[:, -2])

    @staticmethod
    def acc_rev_(A, X):
        if X.size(1) == 1:
            return
        T = 2 * (X.size(1) // 2)
        Aa = A[:, -T:].view(A.size(0), T // 2, 2, -1)
        Xa = X[:, -T:].view(X.size(0), T // 2, 2, -1)
        Xa[:, :, 0].add_(Aa[:, :, 1].mul(Xa[:, :, 1]))
        B = Aa[:, :, 0].clone()
        B[:, 1:].mul_(Aa[:, :-1, 1])
        PScan.acc_rev_(B, Xa[:, :, 0])
        Xa[:, :-1, 1].add_(Aa[:, 1:, 0].mul(Xa[:, 1:, 0]))
        if T < A.size(1):
            X[:, 0].add_(A[:, 1].mul(X[:, 1]))

    # A is NxT, X is NxTxD, Y_init is NxD
    #
    # returns Y of same shape as X, with
    #
    # Y[:, t] = A[:, 0] * Y_init   + X[:, 0] if t == 0
    #         = A[:, t] * Y[:, t-1] + X[:, t] otherwise

    @staticmethod
    def forward(ctx, A, X, Y_init):
        ctx.A = A[:, :, None].clone()
        ctx.Y_init = Y_init[:, None, :].clone()
        ctx.A_star = ctx.A.clone()
        ctx.X_star = X.clone()
        PScan.expand_(ctx.A_star, ctx.X_star)
        return ctx.A_star * ctx.Y_init + ctx.X_star

    @staticmethod
    def backward(ctx, grad_output):
        # ppprint(grad_output)
        U = grad_output * ctx.A_star
        A = ctx.A.clone()
        R = grad_output.clone()
        PScan.acc_rev_(A, R)
        Q = ctx.Y_init.expand_as(ctx.X_star).clone()
        Q[:, 1:].mul_(ctx.A_star[:, :-1]).add_(ctx.X_star[:, :-1])
        return (Q * R).sum(-1), R, U.sum(dim=1)

pscan = PScan.apply



class LRU(nn.Module):  # Implements a Linear Recurrent Unit (LRU) following the parametrization of
# the paper " Resurrecting Linear Recurrences ".
# The LRU is simulated using Parallel Scan (fast!) when "scan" is set to True (default), otherwise recursively (slow).
    def __init__(self, in_features, out_features, state_features, scan = True, rmin=0.9, rmax=1, max_phase=6.283):
        super().__init__()
        self.state_features = state_features
        self.in_features = in_features
        self.scan = scan
        self.out_features = out_features
        self.D = nn.Parameter(torch.randn([out_features, in_features]) / math.sqrt(in_features))
        u1 = torch.rand(state_features)
        u2 = torch.rand(state_features)
        self.nu_log = nn.Parameter(torch.log(-0.5 * torch.log(u1 * (rmax + rmin) * (rmax - rmin) + rmin ** 2)))
        self.theta_log = nn.Parameter(torch.log(max_phase * u2))
        Lambda_mod = torch.exp(-torch.exp(self.nu_log))
        self.gamma_log = nn.Parameter(torch.log(torch.sqrt(torch.ones_like(Lambda_mod) - torch.square(Lambda_mod))))
        B_re = torch.randn([state_features, in_features]) / math.sqrt(2 * in_features)
        B_im = torch.randn([state_features, in_features]) / math.sqrt(2 * in_features)
        self.B = nn.Parameter(torch.complex(B_re, B_im))
        C_re = torch.randn([out_features, state_features]) / math.sqrt(state_features)
        C_im = torch.randn([out_features, state_features]) / math.sqrt(state_features)
        self.C = nn.Parameter(torch.complex(C_re, C_im))
        self.register_buffer('state', torch.complex(torch.zeros(state_features), torch.zeros(state_features)))




    def forward(self, input):
        self.state = self.state
        Lambda_mod = torch.exp(-torch.exp(self.nu_log))
        Lambda_re = Lambda_mod * torch.cos(torch.exp(self.theta_log))
        Lambda_im = Lambda_mod * torch.sin(torch.exp(self.theta_log))
        Lambda = torch.complex(Lambda_re, Lambda_im)  # Eigenvalues matrix
        gammas = torch.exp(self.gamma_log).unsqueeze(-1)
        output = torch.empty([i for i in input.shape[:-1]] + [self.out_features], device=self.B.device)
        # Input must be (Batches,Seq_length, Input size), otherwise adds dummy dimension = 1 for batches
        if input.dim() == 2:
            input = input.unsqueeze(0)


        if self.scan: # Simulate the LRU with Parallel Scan
            input = input.permute(2, 1, 0)  # (Input size,Seq_length, Batches)
            # Unsqueeze b to make its shape (N, V, 1, 1)
            B_unsqueezed = self.B.unsqueeze(-1).unsqueeze(-1)
            # Now broadcast b along dimensions T and D so it can be multiplied elementwise with u
            B_broadcasted = B_unsqueezed.expand(self.state_features, self.in_features, input.shape[1], input.shape[2])
            # Expand u so that it can be multiplied along dimension N, resulting in shape (N, V, T, D)
            input_broadcasted = input.unsqueeze(0).expand(self.state_features, self.in_features, input.shape[1], input.shape[2])
            # Elementwise multiplication and then sum over V (the second dimension)
            inputBU = torch.sum(B_broadcasted * input_broadcasted, dim=1) # (State size,Seq_length, Batches)

            # Prepare matrix Lambda for scan
            Lambda = Lambda.unsqueeze(1)
            A = torch.tile(Lambda, (1, inputBU.shape[1]))
            # Initial condition
            init = torch.complex(torch.zeros((self.state_features, inputBU.shape[2]),  device = self.B.device),
                                 torch.zeros((self.state_features, inputBU.shape[2]),  device = self.B.device))

            gammas_reshaped = gammas.unsqueeze(2)  # Shape becomes (State size, 1, 1)
            # Element-wise multiplication
            GBU = gammas_reshaped * inputBU


            states = pscan(A, GBU, init) # dimensions: (State size,Seq_length, Batches)

            # Prepare output matrices C and D for sequence and batch handling
            # Unsqueeze C to make its shape (Y, X, 1, 1)
            C_unsqueezed = self.C.unsqueeze(-1).unsqueeze(-1)
            # Now broadcast C along dimensions T and D so it can be multiplied elementwise with X
            C_broadcasted = C_unsqueezed.expand(self.out_features, self.state_features, inputBU.shape[1], inputBU.shape[2])
            # Elementwise multiplication and then sum over V (the second dimension)
            CX = torch.sum(C_broadcasted * states, dim=1)

            # Unsqueeze D to make its shape (Y, U, 1, 1)
            D_unsqueezed = self.D.unsqueeze(-1).unsqueeze(-1)
            # Now broadcast C along dimensions T and D so it can be multiplied elementwise with X
            D_broadcasted = D_unsqueezed.expand(self.out_features, self.in_features, input.shape[1], input.shape[2])
            # Elementwise multiplication and then sum over V (the second dimension)
            DU = torch.sum(D_broadcasted * input, dim=1)

            output = 2 * CX.real + DU
            output = output.permute(2, 1, 0)  # Back to (Batches, Seq length, Input size)
        else: # Simulate the LRU recursively
            for i, batch in enumerate(input):
                out_seq = torch.empty(input.shape[1], self.out_features)
                for j, step in enumerate(batch):
                    self.state = (Lambda * self.state + gammas * self.B @ step.to(dtype=self.B.dtype))
                    out_step =  (self.C @ self.state).real + self.D @ step
                    out_seq[j] = out_step
                self.state = torch.complex(torch.zeros_like(self.state.real), torch.zeros_like(self.state.real))
                output[i] = out_seq
        return output # Shape (Batches,Seq_length, Input size)





class SSM(nn.Module):  # Implements LRU + a user-defined scaffolding, this is our SSM block.
    # Scaffolding can be modified. In this case we have LRU, MLP plus linear skip connection.
    def __init__(self, in_features, out_features, state_features, scan, mlp_hidden_size=30, rmin=0.9, rmax=1,
                 max_phase=6.283):
        super().__init__()
        self.mlp = MLP(out_features, mlp_hidden_size, out_features)
        self.LRUR = LRU_Robust(in_features, out_features, state_features, scan, rmin, rmax, max_phase)
        self.model = nn.Sequential(self.LRUR, self.mlp)
        self.lin = nn.Linear(in_features, out_features, bias=false)


    def set_paramS(self):
        self.LRUR.set_param()

    def forward(self, input):
        result = self.model(input) + self.lin(input)
        return result


class DeepLRU(nn.Module):  # Implements a cascade of N SSMs. Linear pre- and post-processing can be modified
    def __init__(self, N, in_features, out_features, mid_features, state_features, scan = True):
        super().__init__()
        self.linin = nn.Linear(in_features, mid_features, bias=false)
        self.linout = nn.Linear(mid_features, out_features, bias=false)
        self.modelt = nn.ModuleList(
            [SSM(mid_features, mid_features, state_features, scan) for j in range(N)])
        self.modelt.insert(0, self.linin)
        self.modelt.append(self.linout)
        self.model = nn.Sequential(*self.modelt)
        self.set_param()

    def set_param(self):
        # Apply the 'custom_method' to all elements except the first and last
        for i in range(1, len(self.modelt) - 1):
            if isinstance(self.modelt[i], SSM):  # Check if it's an instance of CustomModule
                self.modelt[i].set_paramS()  # Call the custom method
    def forward(self, input):
        result = self.model(input)
        return result


# WORK IN PROGRESS

class LRU_Robust(nn.Module):  # Implements a Linear Recurrent Unit (LRU) with trainable l2 gain gamma.
# The LRU is simulated using Parallel Scan (fast!) when "scan" is set to True (default), otherwise recursively (slow).
    def __init__(self, in_features, out_features, state_features, scan = True, rmin=0.9, rmax=1, max_phase=6.283):
        super().__init__()
        self.state_features = state_features
        self.in_features = in_features
        self.out_features = out_features
        self.scan = scan
        self.out_features = out_features
        #Lambda_mod = torch.exp(-torch.exp(self.nu_log))
        #self.gamma_log = nn.Parameter(torch.log(torch.sqrt(torch.ones_like(Lambda_mod) - torch.square(Lambda_mod))))
        self.register_buffer('state', torch.complex(torch.zeros(state_features), torch.zeros(state_features)))


        self.alpha= nn.Parameter(torch.randn(1,1))
        self.beta = nn.Parameter(torch.randn(1, 1))
        self.gamma = nn.Parameter(torch.randn(1, 1)) # l2 gain

        self.epsilon = 0.4
        #self.register_buffer('B', torch.complex(torch.zeros(state_features,in_features), torch.zeros(state_features, in_features)))
        #self.register_buffer('D', torch.complex(torch.zeros(in_features, in_features), torch.zeros(in_features, in_features)))
        self.register_buffer('ID', torch.eye(state_features))
        self.register_buffer('IDu', torch.eye(in_features))

        self.Skew = nn.Parameter(torch.randn(state_features, state_features))
        # self.H21 = nn.Parameter(torch.randn(state_features, state_features))
        # self.H22 = self.g*self.ID # Make H22 a multiple of the identity
        # self.H23 = nn.Parameter(torch.randn(state_features, state_features))

        # Define each block of X as a parameter
        self.X11 = nn.Parameter(torch.randn(state_features, state_features))
        self.X12 = nn.Parameter(torch.randn(state_features, in_features))
        self.X22 = nn.Parameter(torch.randn(in_features, in_features))
        self.X21 = nn.Parameter(torch.randn(in_features, state_features))

        self.C = nn.Parameter(torch.randn(out_features, state_features))
        self.D = nn.Parameter(torch.randn(out_features, in_features))

    def set_param(self):  # Parameter update for L2 gain (free param)

        a = torch.sigmoid(self.alpha)*torch.sigmoid(self.beta)*self.gamma**2
        b = (1-torch.sigmoid(self.alpha))*torch.sigmoid(self.beta)*self.gamma**2
        c = torch.sigmoid(self.alpha)*(1-torch.sigmoid(self.beta))*self.gamma**2
        d = (1-torch.sigmoid(self.alpha))*(1-torch.sigmoid(self.beta))*self.gamma**2

        # Spectral norms of X21, X22 and D
        norm_X21 = torch.linalg.norm(self.X21, ord=2)
        norm_X22 = torch.linalg.norm(self.X22, ord=2)
        norm_D = torch.linalg.norm(self.D, ord=2)

        # Define the modified blocks based on the given constraints
        X21n = (torch.sqrt(a) / norm_X21) * self.X21
        X22n = (torch.sqrt(b) / norm_X22) * self.X22
        Dn = (torch.sqrt(c) / norm_D) * self.D

        # Create a skew-symmetric matrix
        Sk = self.Skew - self.Skew.T
        # Create orthogonal matrix via Cayley Transform
        Q = (self.ID-Sk)@torch.linalg.inv(self.ID+Sk)


        # Compute the blocks of H= X*X.T
        HHt_11 = self.X11 @ self.X11.T + self.X12@self.X12.T+self.C.T@self.C
        HHt_12 = self.X11 @ X21n.T + self.X12 @ X22n.T + self.C.T@Dn
        HHt_21 = HHt_12.T
        HHt_22 = X21n @ X21n.T + X22n @ X22n.T +Dn.T@Dn

        # # Assemble H*H.T in block form
        # HHt = torch.cat([
        #     torch.cat([HHt_11, HHt_12], dim=1),
        #     torch.cat([HHt_21, HHt_22], dim=1)
        # ], dim=0)


        V = HHt_22-self.gamma**2*self.IDu
        R = HHt_12@torch.linalg.inv(V).T@HHt_12.T
        # L, U = torch.linalg.eigh(R)
        # L2, U2 = torch.linalg.eigh(R-HHt_11)

        CR = torch.linalg.cholesky(-R)
        CRH = torch.linalg.cholesky(-R+HHt_11)

        Atilde = CRH@Q@torch.linalg.inv(CR)

        A = torch.linalg.inv(Atilde).T
        self.P = -Atilde@R@Atilde.T
        la= torch.abs(torch.linalg.eigvals(A))
        #lp = torch.linalg.eigvals(self.P)
        B = torch.linalg.pinv(HHt_12.T@Atilde.T)@V.T
        self.B = B
        self.LambdaM = A







        # row1 = torch.cat([-self.LambdaM.T@self.P@ self.LambdaM+self.P, -self.LambdaM.T@self.P@self.B], dim=1)
        # row2 = torch.cat([-(self.LambdaM.T@self.P@self.B).T, -self.B.T@self.P@self.B+self.gamma**2*self.IDu], dim=1)
        # M = torch.cat([row1, row2], dim=0)
        # eigs = torch.linalg.eigvals(M)
        #
        # eigs

    def forward(self, input):
        self.state = self.state.real
        output = torch.empty([i for i in input.shape[:-1]] + [self.out_features], device=self.B.device)
        # Input must be (Batches,Seq_length, Input size), otherwise adds dummy dimension = 1 for batches
        if input.dim() == 2:
            input = input.unsqueeze(0)



        if self.scan: # Simulate the LRU with Parallel Scan
            input = input.permute(2, 1, 0)  # (Input size,Seq_length, Batches)
            # Unsqueeze b to make its shape (N, V, 1, 1)
            B_unsqueezed = self.B.unsqueeze(-1).unsqueeze(-1)
            # Now broadcast b along dimensions T and D so it can be multiplied elementwise with u
            B_broadcasted = B_unsqueezed.expand(self.state_features, self.in_features, input.shape[1], input.shape[2])
            # Expand u so that it can be multiplied along dimension N, resulting in shape (N, V, T, D)
            input_broadcasted = input.unsqueeze(0).expand(self.state_features, self.in_features, input.shape[1], input.shape[2])
            # Elementwise multiplication and then sum over V (the second dimension)
            inputBU = torch.sum(B_broadcasted * input_broadcasted, dim=1) # (State size,Seq_length, Batches)

            # Prepare matrix Lambda for scan
            Lambda = self.Lambda.unsqueeze(1)
            A = torch.tile(Lambda, (1, inputBU.shape[1]))
            # Initial condition
            init = torch.complex(torch.zeros((self.state_features, inputBU.shape[2]),  device = self.B.device),
                                 torch.zeros((self.state_features, inputBU.shape[2]),  device = self.B.device))

            # gammas_reshaped = gammas.unsqueeze(2)  # Shape becomes (State size, 1, 1)
            # Element-wise multiplication
            GBU = inputBU


            states = pscan(A, GBU, init) # dimensions: (State size,Seq_length, Batches)

            # Prepare output matrices C and D for sequence and batch handling
            # Unsqueeze C to make its shape (Y, X, 1, 1)
            C_unsqueezed = self.Cc.unsqueeze(-1).unsqueeze(-1)
            # Now broadcast C along dimensions T and D so it can be multiplied elementwise with X
            C_broadcasted = C_unsqueezed.expand(self.out_features, self.state_features, inputBU.shape[1], inputBU.shape[2])
            # Elementwise multiplication and then sum over V (the second dimension)
            CX = torch.sum(C_broadcasted * states, dim=1)

            # Unsqueeze D to make its shape (Y, U, 1, 1)
            D_unsqueezed = self.D.unsqueeze(-1).unsqueeze(-1)
            # Now broadcast C along dimensions T and D so it can be multiplied elementwise with X
            D_broadcasted = D_unsqueezed.expand(self.out_features, self.in_features, input.shape[1], input.shape[2])
            # Elementwise multiplication and then sum over V (the second dimension)
            DU = torch.sum(D_broadcasted * input, dim=1)

            output = CX.real + DU
            output = output.permute(2, 1, 0)  # Back to (Batches, Seq length, Input size)
        else: # Simulate the LRU recursively
            for i, batch in enumerate(input):
                out_seq = torch.empty(input.shape[1], self.out_features)
                for j, step in enumerate(batch):
                    self.state = (self.LambdaM @ self.state + self.B @ step.to(dtype=self.B.dtype))
                    out_step = (self.C @ self.state+ self.D @ step).real
                    out_seq[j] = out_step
                self.state = torch.zeros_like(self.state.real)
                output[i] = out_seq
        return output # Shape (Batches,Seq_length, Input size)
