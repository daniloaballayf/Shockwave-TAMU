'''
Module has implementation of exact solution for the Euler's Equations.
'''

import numpy as np
from solver.utils import _exact_sol, _compute_gamma_constants


def ext_riem_solver(x, x0, rho, u, p, e, dt, Nt, gamma=1.4, NRITER=20, TOLPRE=1.0E-06):
    '''
    Exact solver for Riemann problem.

    Parameters
    ---------------------
    x: np.ndarray
        A grid of the domain.
        Size Nx X 1.
    x0: float
        Place of discontinuity.
    rho: np.ndarray
        An array with the initial conditions for the density
        evaluated in the domain.
        Size Nx X 1.
    u: np.ndarray
        An array of the initial velocity evaluated in the domain.
        Size Nx X 1.
    p: np.ndarray
        An array of the initial pressure evaluated in the domain.
        Size Nx X 1.
    e: np.ndarray
        An array of the initial total energy evaluated in the domain.
        Size Nx X 1.
    dt: float
        Size of the timestep.
    Nt: int
        Number of timesteps.
    gamma: float
        Heat Capacity Ratio.
    NRITER: int
        Max number of iterations for iterative solver.
        Default is 20.
    TOLPRE: float
        Minimum tolerance for iterative solver.
        Default is 1.0e-6
    
    Returns
    ---------------------
    ext_den: np.ndarray
        An array of shape (Nx X Nt) with the solution for density
        in the domain and timesteps.
    ext_vel: np.ndarray
        An array of shape (Nx X Nt) with the solution for velocity
        in the domain and timesteps.
    ext_pre: np.ndarray
        An array of shape (Nx X Nt) with the solution for pressure
        in the domain and timesteps.
    ext_ene: np.ndarray
        An array of shape (Nx X Nt) with the solution for internal energy
        in the domain and timesteps.
    '''
    Nx = x.shape[0]
    dx = x[1] - x[0]
    L = x[-1]

    DL, UL, PL, DR, UR, PR = rho[0], u[0], p[0], rho[-1], u[-1], p[-1]
    G1, G2, G3, G4, G5, G6, G7, G8 = _compute_gamma_constants(gamma)

    CL, CR = np.sqrt(gamma*PL/DL), np.sqrt(gamma*PR/DR) 

    ext_den = np.zeros((Nx, Nt+1))
    ext_vel = np.zeros((Nx, Nt+1))
    ext_pre = np.zeros((Nx, Nt+1))
    ext_ene = np.zeros((Nx, Nt+1))

    ext_den[:, 0] = rho 
    ext_vel[:, 0] = u
    ext_pre[:, 0] = p 
    ext_ene[:, 0] = (e/rho - 0.5*u**2)

    for i in range(1, Nt+1):
        rho_t, u_t, p_t, e_t = _exact_sol(dt*i, Nx, dx, gamma,
                                          CL, CR, DL, DR, UL, UR, PL, PR,
                                          G1, G2, G3, G4, G5, G6, G7, G8, x0,
                                          NRITER=NRITER, TOLPRE=TOLPRE)
        ext_den[:, i] = rho_t
        ext_vel[:, i] = u_t
        ext_pre[:, i] = p_t 
        ext_ene[:, i] = e_t
    
    return ext_den, ext_vel, ext_pre, ext_ene
