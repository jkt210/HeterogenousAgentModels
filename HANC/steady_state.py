import time
import numpy as np

from consav.grids import equilogspace
from consav.markov import log_rouwenhorst
from consav.misc import elapsed
import scipy.optimize

def prepare_hh_ss(model):
    """ prepare the household block to solve for steady state """

    par = model.par
    ss = model.ss

    ############
    # 1. grids #
    ############
    
    # a. a
    par.a_grid[:] = equilogspace(0.00001,par.a_max,par.Na)
    
    # b. z
    par.z_grid[:],z_trans,z_ergodic,_,_ = log_rouwenhorst(par.rho_z, par.sigma_psi, par.Nz)

    #############################################
    # 2. transition matrix initial distribution #
    #############################################
    
    ss.z_trans[0,:,:] = z_trans
    ss.Dz[0,:] = z_ergodic
    ss.Dbeg[0,:,0] = ss.Dz[0,:] # ergodic at a_lag = 0.0
    ss.Dbeg[0,:,1:] = 0.0 # none with a_lag > 0.0

    ################################################
    # 3. initial guess for intertemporal variables #
    ################################################

    # a. raw value
    y = ss.w*par.zeta*par.z_grid
    c = m = ss.r*(1 - par.tau_a)*par.a_grid[np.newaxis,:] + y[:,np.newaxis]
    v_a = (1+ss.r)*c**(-par.sigma)

    # b. expectation
    ss.vbeg_a[:] = ss.z_trans@v_a
    
def obj_ss(KL,model,do_print=False):
    """ objective when solving for steady state capital """

    par = model.par
    ss = model.ss

    # a. prices
    ss.rk = par.alpha*par.Gamma*(KL)**(par.alpha-1.0)
    ss.w = (1.0-par.alpha)*par.Gamma*(KL)**par.alpha
    ss.r = ss.rB = ss.rk - par.delta

    # b. households
    model.solve_hh_ss(do_print=do_print)
    model.simulate_hh_ss(do_print=do_print)

    # c. government
    ss.B = (ss.tau_a*ss.r*ss.A_hh + ss.tau_l*ss.w*ss.L_hh - par.G_ss)/ss.rB
    
    # d. market clearing
    ss.L = ss.L_hh
    ss.K = KL*ss.L

    ss.clearing_L = ss.L - ss.L_hh
    ss.clearing_A = ss.A_hh - ss.K - ss.B # the asset markov clearing condition
    
    return ss.clearing_A

def find_ss(model,do_print=False,tol=1e-8):
    """ find steady state using the direct or indirect method """

    t0 = time.time()

    par = model.par
    ss = model.ss

    # add bound for KL
    min_KL = (par.delta/par.alpha*par.Gamma)**(1/(par.alpha-1))
    max_KL = ((1/par.beta-1+par.delta)/par.alpha*par.Gamma)**(1/(par.alpha-1))

    res = scipy.optimize.root_scalar(obj_ss,bracket=(min_KL,max_KL),args=(model),method = 'brentq')

    if do_print: print(f'found steady state in {elapsed(t0)}')