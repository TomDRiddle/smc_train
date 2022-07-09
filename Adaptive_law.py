from Differentiator import NTD_m
class Adaptive_Law:
    def __init__(self,t_sampling):
        self.t_sampling = t_sampling
        self.F = 0
        self.F_hat = 0
        self.F0_TD = NTD_m(t_sampling=self.t_sampling)
        self.U_TD = NTD_m(t_sampling=self.t_sampling)
        self.A_TD = NTD_m(t_sampling=self.t_sampling)

    def dF_hat_law(self,ddx,dF0,S,dU,B0=1):
        return ddx-dF0-S-B0*dU

    def Update(self,F0,A,U,S,B0):
        dF0,_ = self.F0_TD.Update(F0)
        dA,_ = self.A_TD.Update(A)
        dU,_ = self.U_TD.Update(U)
        dF_hat = self.dF_hat_law(dA,dF0,S,dU,B0)
        self.F_hat=self.F_hat+dF_hat*self.t_sampling
        return self.F_hat
