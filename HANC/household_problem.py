import numpy as np
import numba as nb

from consav.linear_interp import interp_1d_vec

@nb.njit(parallel=True)        
def solve_hh_backwards(par,z_trans,r,w,tau_a,tau_l,vbeg_a_plus,vbeg_a,a,c,u,ell):
    
    tol_ell = 1e-8
    max_iter_ell = 10000

    """ solve backwards with vbeg_a from previous iteration (here vbeg_a_plus) """

    # taxes
    r = r * (1 - tau_a)
    w = w * (1 - tau_l)
    
    for i_fix in nb.prange(par.Nfix):

        # a. solve step
        for i_z in range(par.Nz):
            
            # a. prepare
            fac = ((w*par.z_grid[i_z]*par.zeta)/par.varphi)**(1/par.nu)
            
            # b. EGM
            c_endo = (par.beta*vbeg_a_plus[i_fix,i_z])**(-1/par.sigma)
            ell_endo = fac*(c_endo)**(-par.sigma/par.nu)
            m_endo = c_endo + par.a_grid - w*ell_endo*par.z_grid[i_z]*par.zeta
            
            # c. interpolation
            m_exo = (1+r)*par.a_grid
            c[i_fix,i_z] = np.zeros(par.Na)
            interp_1d_vec(m_endo, c_endo, m_exo, c[i_fix,i_z])
            ell[i_fix,i_z] = np.zeros(par.Na)
            interp_1d_vec(m_endo, ell_endo, m_exo, ell[i_fix,i_z])
            a[i_fix,i_z] = m_exo + w*ell[i_fix,i_z]*par.z_grid[i_z]*par.zeta - c[i_fix,i_z]
            u[i_fix,i_z] = c[i_fix,i_z]**(1 - par.sigma)/(1 - par.sigma) - par.varphi*ell[i_fix,i_z]**(1 + par.nu)/(1 + par.nu)
            
            # d. refinement at borrowing constraint
            for i_a in range(par.Na):

                if a[i_fix,i_z,i_a] < 0.0:
                    
                    # i. binding constraint for a
                    a[i_fix,i_z,i_a] = 0.0
                    
                    # ii. solve FOC for ell
                    elli = ell[i_fix,i_z,i_a]
                    
                    it = 0
                    while True:
                        
                        ci = (1+r)*par.a_grid[i_a] + w*elli*par.z_grid[i_z]*par.zeta
                        
                        error = elli - fac*ci**(-par.sigma/par.nu)
                        if np.abs(error) < tol_ell:
                            break
                        else:
                            derror = 1 - fac*(-par.sigma/par.nu)*ci**(-par.sigma/par.nu-1)*w*par.z_grid[i_z]*par.zeta
                            elli = elli - error/derror
                        
                        it += 1
                        if it > max_iter_ell: raise ValueError('too many iterations')
                    
                    # iii. save
                    c[i_fix,i_z,i_a] = ci
                    ell[i_fix,i_z,i_a] = elli

        # b. expectation step
        v_a = (1+r)*c[i_fix]**(-par.sigma)
        vbeg_a[i_fix] = z_trans[i_fix]@v_a