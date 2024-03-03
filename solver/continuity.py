'''
Module has implementations of schemes for solving the continuity equation.
'''

import numpy as np
import scipy.sparse as sp

def FCT(x, u, rho_0, dt, Nt, BC='transitive'):
    '''
    FCT method implementation.

    Parameters
    ---------------------
    x: np.ndarray
        A grid of the domain.
        Size Nx X 1.
    u: np.ndarray
        An array of the velocity evaluated in the domain.
        Size Nx X 1.
    rho_0: np.ndarray
        An array with the initial conditions for the density
        evaluated in the domain.
        Size Nx X 1.
    dt: float
        Size of the timestep.
    Nt: int
        Number of timesteps.
    BC: string
        Boundary conditions for the problem. 
        Implemented 'transitive' and 'periodic'.
        Default is 'transitive'.
    
    Returns
    ---------------------
    solution: np.ndarray
        An array of shape (Nx X Nt) with the solution in the domain
        and timesteps.
    '''
    Nx = x.shape[0]
    dx = x[1] - x[0]

    if BC.lower() == 'transitive':
        eps = np.zeros(Nx-1)
        eps[:] = (u[1:] + u[:-1]) * dt / (2*dx)
        nu = 0.5 * np.abs(eps) 

        rho_tilde = np.zeros(Nx)
        mu = nu - 0.5*np.abs(eps)
        solution = np.zeros((Nx, Nt+1))
        solution[:, 0] = rho_0[:]

        d_1 = np.zeros(Nx-1)
        d_2 = np.zeros(Nx-1)
        d_p = np.ones(Nx)
        d_1[1:] = nu[1:] - 0.5*eps[1:]
        d_2[:-1] = nu[:-1] + 0.5*eps[:-1]
        d_p[1:-1] = 1 - 0.5*(eps[1:] - eps[:-1]) - nu[:-1] - nu[1:]
        M = sp.diags([d_2, d_p, d_1], offsets=[-1, 0, 1], shape = (Nx,Nx), format='csr')
        for i in range(1, Nt+1):
            rho_tilde = M@solution[:,i-1]
            f_ad = mu*(rho_tilde[1:] - rho_tilde[:-1])
            s = np.sign(rho_tilde[1:] - rho_tilde[:-1])
            aux = np.zeros(Nx-1)
            aux[:-1] = (rho_tilde[2:] - rho_tilde[1:-1])
            f_cor = s * np.maximum(0, np.minimum(s*aux, np.abs(f_ad), s*(rho_tilde[1:] - rho_tilde[:-1])))
            solution[1:-1, i] = rho_tilde[1:-1] - f_cor[1:] + f_cor[:-1]
            solution[0, i] = solution[1, i]
            solution[-1, i] = solution[-2,i]
    
    if BC.lower() == 'periodic':
        eps = np.zeros(Nx)
        eps[:-1] = (u[1:] + u[:-1]) * dt / (2*dx)
        eps[-1] = (u[0] + u[-1]) * dt / (2*dx)
        nu = 0.5 * np.abs(eps) + 0.15

        d_1 = np.zeros(Nx-1)
        d_2 = np.zeros(Nx-1)
        d_p = np.ones(Nx)
        d_1[:] = nu[:-1] - 0.5*eps[:-1]
        d_2[:] = nu[1:] + 0.5*eps[1:]
        d_p[:-1] = 1 - 0.5*(eps[1:] - eps[:-1]) - nu[:-1] - nu[1:]
        d_p[-1] = 1 - 0.5*(eps[0] - eps[-1]) - nu[-1] - nu[0]
        d_b1 = nu[-1] - 0.5*eps[-1]
        d_b2 = nu[0] + 0.5*eps[0]
        M = sp.diags([d_b1, d_2, d_p, d_1, d_b2], offsets=[-Nx+1, -1, 0, 1, Nx-1], shape = (Nx,Nx), format='csr')

        rho_tilde = np.zeros(Nx)
        f_ad = np.zeros(Nx)
        f_cor = np.zeros(Nx)
        s = np.zeros(Nx)
        aux = np.zeros(Nx)
        difff = np.zeros(Nx)
        mu = nu - 0.5*np.abs(eps)
        solution = np.zeros((Nx, Nt+1))
        solution[:, 0] = rho_0

        for i in range(1, Nt+1):
            rho_tilde = M@solution[:,i-1]
            f_ad[:-1] = mu[:-1]*(rho_tilde[1:] - rho_tilde[:-1])
            f_ad[-1] = mu[-1]*(rho_tilde[0] - rho_tilde[-1])
            s[:-1] = np.sign(rho_tilde[1:] - rho_tilde[:-1])
            s[-1] = np.sign(rho_tilde[0] - rho_tilde[-1])
            aux[:-2] = (rho_tilde[2:] - rho_tilde[1:-1])
            aux[-2] = (rho_tilde[0] - rho_tilde[-1])
            aux[-1] = (rho_tilde[1] - rho_tilde[0])
            difff[:-1] = (rho_tilde[1:] - rho_tilde[:-1])
            difff[-1] = (rho_tilde[0] - rho_tilde[-1])
            f_cor = s * np.maximum(0, np.minimum(s*aux, np.abs(f_ad), s*difff))
            
            solution[:-1, i] = rho_tilde[:-1] - f_cor[1:] + f_cor[:-1]
            solution[-1, i] = rho_tilde[-1] - f_cor[0] + f_cor[-1]

    else:
        print('Not implemented')

    return solution


def laxwendroff(x, u, rho_0, dt, Nt):
    '''
    Lax-Wendroff method implementation.

    Parameters
    ---------------------
    x: np.ndarray
        A grid of the domain.
        Size Nx X 1.
    u: np.ndarray
        An array of the velocity evaluated in the domain.
        Size Nx X 1.
    rho_0: np.ndarray
        An array with the initial conditions for the density
        evaluated in the domain.
        Size Nx X 1.
    dt: float
        Size of the timestep.
    Nt: int
        Number of timesteps.
    
    Returns
    ---------------------
    rho_sol: np.ndarray
        An array of shape (Nx X Nt) with the solution in the domain
        and timesteps.
    '''
    Nx = x.shape[0]
    dx = x[1] - x[0]

    rho_sol = np.zeros((Nx, Nt+1))
    rho_sol[:,0] = rho_0
    diff_f = np.zeros(Nx)
    u_mean = 0.5*(u[1:] + u[:-1])

    for i in range(1, Nt+1):
        f = u*rho_sol[:,i-1]

        rho_prov = 0.5*(rho_sol[1:,i-1] + rho_sol[:-1,i-1]) - 0.5*dt*(f[1:] - f[:-1])/dx
        f_prov =  u_mean * rho_prov
        diff_f[1:-1] = f_prov[1:] - f_prov[:-1]
        rho_sol[:,i] = rho_sol[:,i-1] - dt * (diff_f)/dx
    
    return rho_sol
