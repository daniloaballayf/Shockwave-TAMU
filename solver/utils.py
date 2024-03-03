'''
Auxiliar Functions.
'''
import numpy as np

'''
Analytical Solution
'''

def _compute_gamma_constants(GAMMA):
    '''
    Computes auxiliar convenient ratios of the Heat Capacity Ratio.
    '''
    G1 = (GAMMA - 1.0) / (2.0 * GAMMA)
    G2 = (GAMMA + 1.0) / (2.0 * GAMMA)
    G3 = 2.0 * GAMMA / (GAMMA - 1.0)
    G4 = 2.0 / (GAMMA - 1.0)
    G5 = 2.0 / (GAMMA + 1.0)
    G6 = (GAMMA - 1.0) / (GAMMA + 1.0)
    G7 = (GAMMA - 1.0) / 2.0
    G8 = GAMMA - 1.0

    return G1, G2, G3, G4, G5, G6, G7, G8

def _guess_pressure(CL, CR, DL, DR, UL, UR, PL, PR,
                    G1, G2, G3, G4, G5, G6, G7, G8):
    QUSER = 2.0
    CUP = 0.25*(DL + DR)*(CL + CR)
    PPV = max(0.0, 0.5 * (PL + PR) + 0.5 * (UL - UR) * CUP)
    PMIN = min(PL, PR)
    PMAX = max(PL, PR)
    QMAX = PMAX / PMIN
    if QMAX <= QUSER and (PMIN <= PPV and PPV <= PMAX):
        PM = PPV
    else:
        if PPV < PMIN:
            PQ = (PL / PR) ** G1
            UM = (PQ * UL / CL + UR / CR + G4 * (PQ - 1.0)) / (PQ / CL + 1.0 / CR)
            PTL = 1.0 + G7 * (UL - UM) / CL
            PTR = 1.0 + G7 * (UM - UR) / CR
            PM = 0.5 * (PL * PTL ** G3 + PR * PTR ** G3)
        else:
            GEL = np.sqrt((G5 / DL) / (G6 * PL + PPV))
            GER = np.sqrt((G5 / DR) / (G6 * PR + PPV))
            PM = (GEL * PL + GER * PR - (UR - UL)) / (GEL + GER)
    return PM

def _prefun(P, DK, PK, CK, G1, G2, G3, G4, G5, G6, G7, G8):
    if P <= PK:
        PRAT = P / PK
        F = G4 * CK * (PRAT ** G1 - 1.0)
        FD = (1.0 / (DK * CK)) * PRAT ** (-G2)
    else:
        AK = G5 / DK
        BK = G6 * PK
        QRT = np.sqrt(AK / (BK + P))
        F = (P - PK) * QRT
        FD = (1.0 - 0.5 * (P - PK) / (BK + P)) * QRT
    return F, FD

def _starpu(CL, CR, DL, DR, UL, UR, PL, PR,
            G1, G2, G3, G4, G5, G6, G7, G8, 
            NRITER=20, TOLPRE=1.0E-06):

    PSTART = _guess_pressure(CL, CR, DL, DR, UL, UR, PL, PR,
                            G1, G2, G3, G4, G5, G6, G7, G8)
    POLD = PSTART
    UDIFF = UR - UL
    for i in range(1, NRITER + 1):
        FL, FLD = _prefun(POLD, DL, PL, CL, G1, G2, G3, G4, G5, G6, G7, G8)
        FR, FRD = _prefun(POLD, DR, PR, CR, G1, G2, G3, G4, G5, G6, G7, G8)
        P = POLD - (FL + FR + UDIFF) / (FLD + FRD)
        CHANGE = 2.0 * np.abs((P - POLD) / (P + POLD))
        if P < 0.0:
            P = TOLPRE
        POLD = P
        if CHANGE <= TOLPRE:
            break
    U = 0.5 * (UL + UR + FR - FL)
    return P, U

def sample(PM, UM, S, CL, CR, DL, DR, UL, UR, PL, PR,
                G1, G2, G3, G4, G5, G6, G7, G8, GAMMA):
    if S <= UM:
        if PM <= PL:
            SHL = UL - CL
            if S <= SHL:
                D = DL
                U = UL
                P = PL
            else:
                CML = CL*(PM/PL)**G1
                STL = UM - CML
                if S > STL:
                    D = DL*(PM/PL)**(1/GAMMA)
                    U = UM
                    P = PM
                else:
                    U = G5*(CL + G7*UL + S)
                    C = G5*(CL + G7*(UL - S))
                    D = DL*(C/CL)**G4
                    P = PL*(C/CL)**G3
        else:
            PML = PM/PL
            SL = UL - CL*np.sqrt(G2*PML + G1)
            if S <= SL:
                D = DL
                U = UL
                P = PL
            else:
                D = DL*(PML + G6)/(PML*G6 + 1)
                U = UM
                P = PM
    
    else:
        if PM > PR:
            PMR = PM/PR
            SR = UR + CR*np.sqrt(G2*PMR + G1)
            if S >= SR:
                D = DR
                U = UR
                P = PR
            else:
                D = DR*(PMR + G6)/(PMR*G6 + 1)
                U = UM
                P = PM
        else:
            SHR = UR + CR
            if S >= SHR:
                D = DR
                U = UR
                P = PR
            else:
                CMR = CR*(PM/PR)**G1
                STR = UM + CMR
                if S <= STR:
                    D = DR*(PM/PR)**(1/GAMMA)
                    U = UM
                    P = PM
                else:
                    U = G5*(-CR + G7*UR + S)
                    C = G5*(CR - G7*(UR-S))
                    D = DR*(C/CR)**G4
                    P = PR*(C/CR)**G3
    return D, U, P


def _exact_sol(TIMEOUT, Nx, dx, gamma, CL, CR, DL, DR, UL, UR, PL, PR,
                                        G1, G2, G3, G4, G5, G6, G7, G8,
                                        DIAPH, NRITER=20, TOLPRE=1.0E-06):
    '''
    Obtains the exact solution for a time.
    '''
    
    density = np.zeros(Nx)
    pressure = np.zeros(Nx)
    velocity = np.zeros(Nx)
    int_energy = np.zeros(Nx)
    if G4*(CL + CR) <= (UR - UL):
        print('Vacuum')
    else:
        P, U = _starpu(CL, CR, DL, DR, UL, UR, PL, PR, 
                       G1, G2, G3, G4, G5, G6, G7, G8,
                       NRITER=NRITER, TOLPRE=TOLPRE)
        for i in range(1, Nx + 1):
            xpos = (i - 0.5)*dx
            S = (xpos - DIAPH)/TIMEOUT
            DS, US, PS = sample(P, U, S, CL, CR, DL, DR, UL, UR, PL, PR,
                                G1, G2, G3, G4, G5, G6, G7, G8, gamma)
            density[i-1] = DS
            pressure[i-1] = PS
            velocity[i-1] = US
            int_energy[i-1] = PS/DS/G8
    return density, velocity, pressure, int_energy


'''
Roe Solver
'''

def _trrs(ul, ur, al, ar, pl, pr, Q=2, gamma=1.4):
    '''
    Obtains values of speed of sound and velocity for an entropy fix.
    '''
    z = (gamma-1)/(2*gamma)
    num = al + ar - 0.5*(gamma-1)*(ur-ul)
    den = al/pl**z + ar/pr**z
    p_ = (num/den)**(1/z)
    al_, ar_ = al*(p_/pl)**z, ar*(p_/pr)**z
    ul_, ur_ = ul + 2*(al - al_)/(gamma-1), ur + 2*(ar_ - ar)/(gamma-1) 

    return al_, ar_, ul_, ur_

def _roe_flux(rho_l, rho_r, u_l, u_r, e_l, e_r, p_l, p_r, gamma):
    '''
    Obtains the Average Roe Fluxes for Riemann Problem.

    Parameters
    ---------------------
    rho_l: np.ndarray
        Values of the density at the left of discontinuity in Riemann Problem.
    rho_r: np.ndarray
        Values of the density at the right of discontinuity in Riemann Problem.
    u_l: np.ndarray
        Values of the velocity at the left of discontinuity in Riemann Problem.
    u_r: np.ndarray
        Values of the velocity at the right of discontinuity in Riemann Problem.
    e_l: np.ndarray
        Values of the energy at the left of discontinuity in Riemann Problem.
    e_r: np.ndarray
        Values of the energy at the right of discontinuity in Riemann Problem.
    p_l: np.ndarray
        Values of the pressure at the left of discontinuity in Riemann Problem.
    p_r: np.ndarray
        Values of the pressure at the right of discontinuity in Riemann Problem.
    gamma: float
        Heat Capacity Ratio.
    
    Returns
    ---------------------
    flux_rho: np.ndarray
        Roe average Flux for the density.
    flux_mom: np.ndarray
        Roe average Flux for the momentum.
    flux_ene: np.ndarray
        Roe average Flux for the total energy.
    '''
    # Obtain entalpy
    h_l = (e_l + p_l)/rho_l
    h_r = (e_r + p_r)/rho_r

    # Obtain Roe averages
    rho_det = np.sqrt(rho_l) + np.sqrt(rho_r)
    rho_avg = np.sqrt(rho_l*rho_r)
    u_avg = (np.sqrt(rho_l)*u_l + np.sqrt(rho_r)*u_r)/rho_det
    h_avg = (np.sqrt(rho_l)*h_l + np.sqrt(rho_r)*h_r)/rho_det
    a_avg =  np.sqrt((gamma-1)*(h_avg - 0.5*u_avg**2))

    # Obtain eigenvalues
    l_1 = u_avg - a_avg
    l_2 = u_avg
    l_3 = u_avg + a_avg

    # Compute coefs
    alp_1 = 0.5*(1/a_avg**2) * (p_r - p_l - rho_avg*a_avg*(u_r - u_l))
    alp_2 = rho_r - rho_l - (p_r - p_l)/a_avg**2
    alp_3 = 0.5*(1/a_avg**2) * (p_r - p_l + rho_avg*a_avg*(u_r - u_l))

    # Compute eigenvectors
    K1 = np.array([np.ones(u_avg.shape[0]), u_avg - a_avg, h_avg - u_avg*a_avg])
    K2 = np.array([np.ones(u_avg.shape[0]), u_avg, 0.5*u_avg**2])
    K3 = np.array([np.ones(u_avg.shape[0]), u_avg + a_avg, h_avg + u_avg*a_avg])

    # Entropy fix
    al, ar = np.sqrt(gamma*p_l/rho_l), np.sqrt(gamma*p_r/rho_r)
    al_, ar_, ul_, ur_ = _trrs(u_l, u_r, al, ar, p_l, p_r, gamma=gamma)
    l_1l = u_l - al
    l_1r = ul_ - al_
    l_3l = ur_ + ar_
    l_3r = u_r + ar
    
    idx1 = np.logical_and(l_1l < 0, l_1r > 0)
    idx3 = np.logical_and(l_3l < 0, l_3r > 0)

    idx_t1 = (l_1r - l_1l) != 0
    idx_t3 = (l_3r - l_3l) != 0

    l1_bar = np.zeros(l_1l.shape[0])
    l3_bar = np.zeros(l_1l.shape[0])
    
    l1_bar[idx_t1] = l_1l[idx_t1]*(l_1r[idx_t1] - l_1[idx_t1])/(l_1r[idx_t1] - l_1l[idx_t1])
    l3_bar[idx_t3] = l_3r[idx_t3]*(l_3[idx_t3] - l_3l[idx_t3])/(l_3r[idx_t3] - l_3l[idx_t3])

    # Roe flux
    #idx_body = np.logical_or(np.logical_not(idx1), np.logical_not(idx3))
    flux_rho = 0.5 * (rho_l*u_l + rho_r*u_r)
    flux_rho -= 0.5*(np.abs(l_1)*alp_1*K1[0] + np.abs(l_2)*alp_2*K2[0] + np.abs(l_3)*alp_3*K3[0])
    flux_mom = 0.5 * (rho_l*u_l**2 + p_l + rho_r*u_r**2 + p_r) 
    flux_mom -= 0.5*(np.abs(l_1)*alp_1*K1[1] + np.abs(l_2)*alp_2*K2[1] + np.abs(l_3)*alp_3*K3[1])
    flux_ene = 0.5 * (u_l*(e_l + p_l) + u_r*(e_r + p_r)) 
    flux_ene -= 0.5*(np.abs(l_1)*alp_1*K1[2] + np.abs(l_2)*alp_2*K2[2] + np.abs(l_3)*alp_3*K3[2])

    flux_rho[idx1] = rho_l[idx1] * u_l[idx1] + l1_bar[idx1] * alp_1[idx1]*K1[0][idx1]
    flux_mom[idx1] = rho_l[idx1] * u_l[idx1]**2 + p_l[idx1] + l1_bar[idx1] * alp_1[idx1]*K1[1][idx1]
    flux_ene[idx1] = u_l[idx1]*(e_l[idx1] + p_l[idx1]) + l1_bar[idx1] * alp_1[idx1]*K1[2][idx1]

    flux_rho[idx3] = rho_r[idx3] * u_r[idx3] - l3_bar[idx3] * alp_3[idx3]*K3[0][idx3]
    flux_mom[idx3] = rho_r[idx3] * u_r[idx3]**2 - p_r[idx3] + l3_bar[idx3] * alp_3[idx3]*K3[1][idx3]
    flux_ene[idx3] = u_r[idx3]*(e_r[idx3] + p_r[idx3]) - l3_bar[idx3] * alp_3[idx3]*K3[2][idx3]

    return flux_rho, flux_mom, flux_ene