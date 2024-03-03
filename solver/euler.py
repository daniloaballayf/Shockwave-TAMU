'''
Module has implementations of schemes for solving the Euler's Equations.
'''

import numpy as np
from solver.utils import _roe_flux
import scipy.sparse as sp

def laxwendroff(x, rho, u, p, e, dt, Nt, gamma=1.4):
    '''
    Lax-Wendroff method implementation.

    Parameters
    ---------------------
    x: np.ndarray
        A grid of the domain.
        Size Nx X 1.
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
    
    Returns
    ---------------------
    sol_rho_lw: np.ndarray
        An array of shape (Nx X Nt) with the solution of the density 
        in the domain and timesteps.
    sol_mom_lw: np.ndarray
        An array of shape (Nx X Nt) with the solution of the momentum 
        in the domain and timesteps.
    sol_pre_lw: np.ndarray
        An array of shape (Nx X Nt) with the solution of the pressure
        in the domain and timesteps.
    sol_ene_lw: np.ndarray
        An array of shape (Nx X Nt) with the solution of the total energy
        in the domain and timesteps.
    '''
    Nx = x.shape[0]
    dx = x[1] - x[0]

    sol_rho_lw = np.zeros((Nx, Nt+1))
    sol_mom_lw = np.zeros((Nx, Nt+1))
    sol_ene_lw = np.zeros((Nx, Nt+1))
    sol_pre_lw = np.zeros((Nx, Nt+1))


    sol_rho_lw[:, 0] = rho
    sol_mom_lw[:, 0] = rho * u
    sol_ene_lw[:, 0] = e
    sol_pre_lw[:, 0] = p

    diff_rho = np.zeros(Nx)
    diff_mom = np.zeros(Nx)
    diff_ene = np.zeros(Nx)
    v = np.zeros(Nx)
    v_prov = np.zeros(Nx-1)
    v[sol_rho_lw[:,0]!=0] = sol_mom_lw[:, 0][sol_rho_lw[:,0]!=0]/sol_rho_lw[:,0][sol_rho_lw[:,0]!=0]

    for i in range(1, Nt+1):
        press = sol_pre_lw[:,i-1]
        f_rho = sol_mom_lw[:,i-1]
        f_mom = press + sol_rho_lw[:,i-1]*v**2
        f_ene = v * (sol_ene_lw[:,i-1] + press)
        
        u_prov = 0.5*(sol_mom_lw[1:,i-1] + sol_mom_lw[:-1,i-1]) - 0.5*dt*(f_mom[1:] - f_mom[:-1])/dx
        r_prov = 0.5*(sol_rho_lw[1:,i-1] + sol_rho_lw[:-1,i-1]) - 0.5*dt*(f_rho[1:] - f_rho[:-1])/dx
        e_prov = 0.5*(sol_ene_lw[1:,i-1] + sol_ene_lw[:-1,i-1]) - 0.5*dt*(f_ene[1:] - f_ene[:-1])/dx

        v_prov[r_prov!=0] = u_prov[r_prov!=0]/r_prov[r_prov!=0]
        press_prov = (gamma - 1) * (e_prov - 0.5*r_prov * v_prov**2) 
        f_rho_prov = u_prov
        f_mom_prov = press_prov + r_prov*v_prov**2
        f_ene_prov = v_prov*(e_prov + press_prov)

        diff_rho[1:-1] = f_rho_prov[1:] - f_rho_prov[:-1]
        diff_mom[1:-1] = f_mom_prov[1:] - f_mom_prov[:-1]
        diff_ene[1:-1] = f_ene_prov[1:] - f_ene_prov[:-1]

        sol_mom_lw[:,i] = sol_mom_lw[:,i-1] - dt*(diff_mom)/dx
        sol_rho_lw[:,i] = sol_rho_lw[:,i-1] - dt*(diff_rho)/dx
        sol_ene_lw[:,i] = sol_ene_lw[:,i-1] - dt*(diff_ene)/dx

        sol_mom_lw[0,i] = sol_mom_lw[1,i] 
        sol_rho_lw[0,i] = sol_rho_lw[1,i] 
        sol_ene_lw[0,i] = sol_ene_lw[1,i] 
        sol_mom_lw[-1,i] = sol_mom_lw[-2,i] 
        sol_rho_lw[-1,i] = sol_rho_lw[-2,i] 
        sol_ene_lw[-1,i] = sol_ene_lw[-2,i] 
        
        v[sol_rho_lw[:,i]!=0] = sol_mom_lw[:, i][sol_rho_lw[:,i]!=0]/sol_rho_lw[:,i][sol_rho_lw[:,i]!=0]
        sol_pre_lw[:,i] = (gamma - 1) *(sol_ene_lw[:,i] - 0.5* sol_rho_lw[:,i]* v**2)
    
    return sol_rho_lw, sol_mom_lw, sol_pre_lw, sol_ene_lw

def roe(x, rho, u, p, e, dt, Nt, gamma=1.4):
    '''
    Roe method implementation.

    Parameters
    ---------------------
    x: np.ndarray
        A grid of the domain.
        Size Nx X 1.
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
    
    Returns
    ---------------------
    sol_rho: np.ndarray
        An array of shape (Nx X Nt) with the solution of the density 
        in the domain and timesteps.
    sol_mom: np.ndarray
        An array of shape (Nx X Nt) with the solution of the momentum 
        in the domain and timesteps.
    sol_pre: np.ndarray
        An array of shape (Nx X Nt) with the solution of the pressure
        in the domain and timesteps.
    sol_ene: np.ndarray
        An array of shape (Nx X Nt) with the solution of the total energy
        in the domain and timesteps.
    '''
    Nx = x.shape[0]
    dx = x[1] - x[0]

    sol_rho = np.zeros((Nx, Nt+1))
    sol_mom = np.zeros((Nx, Nt+1))
    sol_ene = np.zeros((Nx, Nt+1))
    sol_pre = np.zeros((Nx, Nt+1))

    v = np.zeros(Nx)
    diff_rho = np.zeros(Nx)
    diff_mom = np.zeros(Nx)
    diff_ene = np.zeros(Nx)

    sol_rho[:, 0] = rho
    sol_mom[:, 0] = rho * u
    sol_ene[:, 0] = e
    sol_pre[:, 0] = p
    v[sol_rho[:,0]!=0] = sol_mom[:, 0][sol_rho[:,0]!=0]/sol_rho[:,0][sol_rho[:,0]!=0]

    for i in range(1, Nt + 1):
        flux_rho, flux_mom, flux_ene = _roe_flux(sol_rho[:-1, i-1], sol_rho[1:, i-1],
                                                v[:-1], v[1:],
                                                sol_ene[:-1, i-1], sol_ene[1:, i-1],
                                                sol_pre[:-1, i-1], sol_pre[1:, i-1],
                                                gamma)
        
        diff_rho[1:-1] =  (flux_rho[:-1] - flux_rho[1:])
        diff_mom[1:-1] =  (flux_mom[:-1] - flux_mom[1:])
        diff_ene[1:-1] =  (flux_ene[:-1] - flux_ene[1:])
        sol_rho[:,i] = sol_rho[:,i-1] + (dt/dx)*diff_rho
        sol_mom[:,i] = sol_mom[:,i-1] + (dt/dx)*diff_mom
        sol_ene[:,i] = sol_ene[:,i-1] + (dt/dx)*diff_ene
        v[sol_rho[:,i]!=0] = sol_mom[:, i][sol_rho[:,i]!=0]/sol_rho[:,i][sol_rho[:,i]!=0]
        sol_pre[:,i] = (gamma - 1) *(sol_ene[:,i] - 0.5* sol_rho[:,i]* v**2)

        # Boundary Condition
        sol_rho[0,i] = sol_rho[1,i]
        sol_mom[0,i] = sol_mom[1,i]
        sol_ene[0,i] = sol_ene[1,i]
        sol_pre[0,i] = sol_pre[1,i]

        sol_rho[-1,i] = sol_rho[-2,i] 
        sol_mom[-1,i] = sol_mom[-2,i] 
        sol_ene[-1,i] = sol_ene[-2,i] 
        sol_pre[-1,i] = sol_pre[-2,i] 
    
    return sol_rho, sol_mom, sol_pre, sol_ene

def FCT(x, rho, u, p, e, dt, Nt, gamma=1.4):
    '''
    FCT method implementation.

    **Implement two step integration**

    Parameters
    ---------------------
    x: np.ndarray
        A grid of the domain.
        Size Nx X 1.
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
    
    Returns
    ---------------------
    rho_sol_fct: np.ndarray
        An array of shape (Nx X Nt) with the solution of the density 
        in the domain and timesteps.
    mom_sol_fct: np.ndarray
        An array of shape (Nx X Nt) with the solution of the momentum 
        in the domain and timesteps.
    pre_sol_fct: np.ndarray
        An array of shape (Nx X Nt) with the solution of the pressure
        in the domain and timesteps.
    ene_sol_fct: np.ndarray
        An array of shape (Nx X Nt) with the solution of the total energy
        in the domain and timesteps.
    '''

    Nx = x.shape[0]
    dx = x[1] - x[0]

    rho_sol_fct = np.zeros((Nx, Nt+1))
    mom_sol_fct = np.zeros((Nx, Nt+1))
    ene_sol_fct = np.zeros((Nx, Nt+1))
    pre_sol_fct = np.zeros((Nx, Nt+1))

    rho_sol_fct[:, 0] = rho
    mom_sol_fct[:, 0] = rho * u 
    ene_sol_fct[:, 0] = e
    pre_sol_fct[:, 0] = p

    v = u

    diff_rho = np.zeros(Nx)
    diff_mom = np.zeros(Nx)
    diff_ene = np.zeros(Nx)

    aux_1_rho = np.zeros(Nx-1)
    aux_1_mom = np.zeros(Nx-1)
    aux_1_ene = np.zeros(Nx-1)

    for i in range(1, Nt+1):
        press = pre_sol_fct[:,i-1]
        f_rho = mom_sol_fct[:,i-1]
        f_mom = press + rho_sol_fct[:,i-1]*v**2
        f_ene = v * (ene_sol_fct[:,i-1] + press)
        
        a = np.sqrt(gamma*pre_sol_fct[:,i-1]/rho_sol_fct[:,i-1])
        a_mean = 0.5*(a[:-1] + a[1:])*dt/dx
        
        f_rho_bb = (dt/dx) * (f_rho[1:] + f_rho[:-1]) * 0.5 - 0.5*((a_mean)**2 + 0.25)*(rho_sol_fct[1:,i-1] - rho_sol_fct[:-1,i-1]) 
        f_mom_bb = (dt/dx) * (f_mom[1:] + f_mom[:-1]) * 0.5 - 0.5*((a_mean)**2 + 0.25)*(mom_sol_fct[1:,i-1] - mom_sol_fct[:-1,i-1]) 
        f_ene_bb = (dt/dx) * (f_ene[1:] + f_ene[:-1]) * 0.5 - 0.5*((a_mean)**2 + 0.25)*(ene_sol_fct[1:,i-1] - ene_sol_fct[:-1,i-1])

        f_ad_rho = 0.125*(rho_sol_fct[1:, i-1] - rho_sol_fct[:-1,i-1])
        f_ad_mom = 0.125*(mom_sol_fct[1:, i-1] - mom_sol_fct[:-1,i-1])
        f_ad_ene = 0.125*(ene_sol_fct[1:, i-1] - ene_sol_fct[:-1,i-1])

        aux_1_rho[:-1] = rho_sol_fct[2:, i-1] - rho_sol_fct[1:-1, i-1]
        aux_1_mom[:-1] = mom_sol_fct[2:, i-1] - mom_sol_fct[1:-1, i-1]
        aux_1_ene[:-1] = ene_sol_fct[2:, i-1] - ene_sol_fct[1:-1, i-1]

        f_rho_c = np.maximum(0, np.minimum(f_ad_rho, aux_1_rho))
        f_mom_c = np.maximum(0, np.minimum(f_ad_mom, aux_1_mom))
        f_ene_c = np.maximum(0, np.minimum(f_ad_ene, aux_1_ene))

        f_rho_bb += f_rho_c
        f_mom_bb += f_mom_c
        f_ene_bb += f_ene_c    
        
        diff_rho[1:-1] = f_rho_bb[1:] - f_rho_bb[:-1] 
        diff_mom[1:-1] = f_mom_bb[1:] - f_mom_bb[:-1]
        diff_ene[1:-1] = f_ene_bb[1:] - f_ene_bb[:-1]

        mom_sol_fct[:,i] = mom_sol_fct[:,i-1] - (diff_mom)
        rho_sol_fct[:,i] = rho_sol_fct[:,i-1] - (diff_rho)
        ene_sol_fct[:,i] = ene_sol_fct[:,i-1] - (diff_ene)

        mom_sol_fct[0,i] = mom_sol_fct[1,i] 
        rho_sol_fct[0,i] = rho_sol_fct[1,i] 
        ene_sol_fct[0,i] = ene_sol_fct[1,i] 
        mom_sol_fct[-1,i] = mom_sol_fct[-2,i] 
        rho_sol_fct[-1,i] = rho_sol_fct[-2,i] 
        ene_sol_fct[-1,i] = ene_sol_fct[-2,i] 
        
        v = mom_sol_fct[:, i]/rho_sol_fct[:,i]
        pre_sol_fct[:,i] = (gamma - 1) *(ene_sol_fct[:,i] - 0.5* rho_sol_fct[:,i]* v**2)

    return rho_sol_fct, mom_sol_fct, pre_sol_fct, ene_sol_fct