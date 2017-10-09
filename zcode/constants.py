"""Common Numerical and Physical Constants.
"""

import numpy as np
import astropy as ap
import astropy.constants
import astropy.cosmology
# from astropy.cosmology import WMAP9 as cosmo
cosmo = astropy.cosmology.WMAP9

# Fundamental Constants
# ---------------------
NWTG = ap.constants.G.cgs.value
SPLC = ap.constants.c.cgs.value
MSOL = ap.constants.M_sun.cgs.value
LSOL = ap.constants.L_sun.cgs.value
RSOL = ap.constants.R_sun.cgs.value
PC = ap.constants.pc.cgs.value
AU = ap.constants.au.cgs.value

ARCSEC = ap.units.arcsec.cgs.scale              # arcsecond in radians

YR = ap.units.year.to(ap.units.s)
DAY = 86400.0                                   # Day in seconds
EVOLT = 1.60218e-12                             # Electronvolt in ergs

MELC = ap.constants.m_e.cgs.value
MPRT = ap.constants.m_p.cgs.value
KBOLTZ = ap.constants.k_B.cgs.value             # Boltzmann constant
HPLANCK = ap.constants.h.cgs.value              # Planck constant
SIGMA_SB = ap.constants.sigma_sb.cgs.value

H0 = cosmo.H0.cgs.value                          # Hubble Constants at z=0.0
HPAR = cosmo.H0.value/100.0
OMEGA_M = cosmo.Om0
OMEGA_B = cosmo.Ob0
OMEGA_DM = cosmo.Odm0
RHO_CRIT = cosmo.critical_density0.cgs.value


# Higher order constants
# ----------------------
# Thomson-Scattering (Electron-Scattering) cross-section
try:
    SIGMA_T = ap.constants.sigma_T.cgs.value
except:
    SIGMA_T = 6.652458734e-25     # cm^2 (i.e. per electron)
# Electron-Scattering Opacity ($\kappa_{es} = n_e \sigma_T / \rho = \mu_e \sigma_T / m_p$)
#     Where $\mu_e$ is the mean-mass per electron, for a total mass-density $\rho$.
KAPPA_ES = SIGMA_T/MPRT


# Derived Constants
# -----------------
PIFT = 4.0*np.pi/3.0                             # (4.0/3.0)*Pi
SCHW = 2*NWTG/(SPLC*SPLC)                        # Schwarzschild Constant (2*G/c^2)
HTAU = 1.0/H0                                    # Hubble Time - 1/H0 [sec]
EDDC = 4.0*np.pi*NWTG*SPLC*MPRT/SIGMA_T          # Eddington Luminosity factor [erg/s/g]

MYR = 1.0e6*YR
GYR = 1.0e9*YR

KPC = 1.0e3*PC
MPC = 1.0e6*PC
GPC = 1.0e9*PC
