import numpy as np
import numba as nb

from GEModelTools import lag, lead
   
@nb.njit
def block_pre(par,ini,ss,path,ncols=1):

    for ncol in range(ncols):

        # unpack
        A = path.A[ncol,:]
        B = path.B[ncol,:]
        chi = path.chi[ncol,:]
        clearing_A = path.clearing_A[ncol,:]
        clearing_Y = path.clearing_Y[ncol,:]
        G = path.G[ncol,:]
        Gamma = path.Gamma[ncol,:]
        i = path.i[ncol,:]
        L = path.L[ncol,:]
        NKWC_res = path.NKWC_res[ncol,:]
        pi_w = path.pi_w[ncol,:]
        pi = path.pi[ncol,:]
        r = path.r[ncol,:]
        tau = path.tau[ncol,:]
        w = path.w[ncol,:]
        Y = path.Y[ncol,:]
        q = path.q[ncol,:]
        ra = path.ra[ncol,:]
        A_hh = path.A_hh[ncol,:]
        C_hh = path.C_hh[ncol,:]

        #################
        # implied paths #
        #################

        # add your own code

        # a. firms
        # guess
        pi_w[:] = 0.0
        Gamma[:] = 1.0

        w[:] = Gamma
        Y[:] = L*Gamma
        Gamma_lag = lag(ini.Gamma, Gamma)
        pi[:] = (1+pi_w)/(Gamma/Gamma_lag) - 1
        
        # b. monetary policy
        for t in range(par.T):
            i_lag = i[t-1] if t > 0 else ini.i
            i[t] = ((1+i_lag)**par.rho_i) * ((1 + ss.r) * (1 + pi[t])**par.phi_pi)**(1 - par.rho_i) - 1
        
        for t in range(par.T):
            pi_plus = pi[t+1] if t < par.T-1 else ss.pi
            r[t] = (1.0+i[t])/(1.0+pi_plus) - 1.0

        # c. government
        q[:] = ss.q
        for t in range(par.T):
            q_plus = q[t+1] if t < par.T-1 else ss.q
            q[t] = (1.0 + par.delta*q_plus)/(1.0 + r[t])

        q_lag = lag(ini.q,q)
        ra[:] = (1.0 + par.delta*q)/q_lag - 1.0

        for t in range(par.T):

            # i. lag
            B_lag = B[t-1] if t > 0 else ini.B
            
            # ii. tau
            tau[t] = ss.tau + par.omega*ss.q*(B_lag - ss.B)/ss.Y
            
            # iii. government debt
            B[t] = (B_lag + G[t] + chi[t] - tau[t]*Y[t])/q[t] + par.delta*B_lag

        # d. aggregates 
        A[:] = q*B

@nb.njit
def block_post(par,ini,ss,path,ncols=1):

    for ncol in range(ncols):

        # unpack
        A = path.A[ncol,:]
        B = path.B[ncol,:]
        chi = path.chi[ncol,:]
        clearing_A = path.clearing_A[ncol,:]
        clearing_Y = path.clearing_Y[ncol,:]
        G = path.G[ncol,:]
        Gamma = path.Gamma[ncol,:]
        i = path.i[ncol,:]
        L = path.L[ncol,:]
        NKWC_res = path.NKWC_res[ncol,:]
        pi_w = path.pi_w[ncol,:]
        pi = path.pi[ncol,:]
        r = path.r[ncol,:]
        tau = path.tau[ncol,:]
        w = path.w[ncol,:]
        Y = path.Y[ncol,:]
        q = path.q[ncol,:]
        ra = path.ra[ncol,:]
        A_hh = path.A_hh[ncol,:]
        C_hh = path.C_hh[ncol,:]
        
        #################
        # check targets #
        #################

        # add your own code

        # a. phillips curve
        for t in range(par.T):
            pi_w_plus = pi_w[t+1] if t < par.T-1 else ss.pi_w
            NKWC_res[t] = pi_w[t] - (par.kappa*(par.varphi*(L[t]**par.nu) - 1.0/par.mu*(1.0 - tau[t])*w[t]*(C_hh[t]**(-par.sigma))) + par.beta*pi_w_plus)

        # b. market clearing
        clearing_A[:] = A-A_hh
        clearing_Y[:] = Y - C_hh - G