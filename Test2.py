import torch


def set_param2(self):  # Parameter update for L2 gain (free param)

    e = torch.abs(self.e)
    g = torch.abs(self.g)
    q = torch.abs(self.q)
    k = torch.abs(self.k)

    # Spectral norms of T21 and T23
    norm_T21 = torch.linalg.norm(self.T21, ord=2)
    norm_T23 = torch.linalg.norm(self.T23, ord=2)

    # Define the modified blocks based on the given constraints
    H21 = (e / norm_T21) * self.T21
    H23 = (k / norm_T23) * self.T23
    H22 = g * self.ID
    H33 = (g + q) * self.ID
    H31 = e * self.ID
    H32 = k * self.ID

    # Compute the blocks of H*H.T
    HHt_11 = self.H11 @ self.H11.T + self.H12 @ self.H12.T + self.H13 @ self.H13.T
    HHt_12 = self.H11 @ H21.T + g * self.H12 + self.H13 @ H23.T
    HHt_13 = e * self.H11 + k * self.H12 + (g + q) * self.H13

    HHt_21 = H21 @ self.H11.T + g * self.H12.T + H23 @ self.H13.T
    HHt_22 = (e ** 2 / norm_T21 ** 2) * (self.T21 @ self.T21.T) + g ** 2 * self.ID + (k ** 2 / norm_T23 ** 2) * (
            self.T23 @ self.T23.T)
    HHt_23 = (e ** 2 / norm_T21) * self.T21 + g * k * self.ID + (k * (g + q) / norm_T23) * self.T23

    HHt_31 = e * self.H11.T + k * self.H12.T + (g + q) * self.H13.T
    HHt_32 = (e ** 2 / norm_T21) * self.T21.T + g * k * self.ID + (k * (g + q) / norm_T23) * self.T23.T
    HHt_33 = e ** 2 * self.ID + k ** 2 * self.ID + (g + q) ** 2 * self.ID

    self.gamma2 = e ** 2 + k ** 2 + (g + q) ** 2

    # Assemble H*H.T in block form
    HHt = torch.cat([
        torch.cat([HHt_11, HHt_12, HHt_13], dim=1),
        torch.cat([HHt_21, HHt_22, HHt_23], dim=1),
        torch.cat([HHt_31, HHt_32, HHt_33], dim=1)
    ], dim=0)

    # Create a skew-symmetric matrix
    Sk = self.Skew - self.Skew.T
    # Create orthogonal matrix via Cayley Transform
    Q = (self.ID - Sk) @ torch.linalg.inv(self.ID + Sk)

    V = HHt_22 - HHt_33
    R = HHt_12 @ torch.linalg.inv(V).T @ HHt_12.T
    L, U = torch.linalg.eigh(R)
    L2, U2 = torch.linalg.eigh(R - HHt_11)

    A = torch.linalg.inv(U2 @ torch.sqrt(torch.abs(torch.diag(L2))) @ Q @
                         torch.linalg.inv(torch.sqrt(torch.abs(torch.diag(L)))) @ U.T).T

    P = -torch.linalg.inv(A).T @ R @ torch.linalg.inv(A)

    la = torch.abs(torch.linalg.eigvals(A))

    lp = torch.linalg.eigvals(P)

    B = A @ torch.linalg.inv(HHt_12).T @ V.T

    self.LambdaM = A
    self.B = B
    self.P = P
    self.C = -HHt_31
    self.D = -HHt_32

    # row1 = torch.cat([-self.LambdaM.T@self.P@ self.LambdaM+self.P, -self.LambdaM.T@self.P@self.B, -self.C.T], dim=1)
    # row2 = torch.cat([-(self.LambdaM.T@self.P@self.B).T, -self.B.T@self.P@self.B+self.gamma2*self.ID, -self.D.T], dim=1)
    # row3 = torch.cat([-self.C, -self.D, self.gamma2*self.ID], dim=1)
    # M = torch.cat([row1, row2, row3], dim=0)
    # eigs = torch.linalg.eigvals(M)