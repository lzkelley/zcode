"""
Common Numerical and Physical Constants.

"""

import numpy as np
import astropy as ap
import astropy.constants
import astropy.cosmology
from astropy.cosmology import WMAP9 as cosmo

NWTG = ap.constants.G.cgs.value
SPLC= ap.constants.c.cgs.value
MSOL = ap.constants.M_sun.cgs.value
LSOL = ap.constants.L_sun.cgs.value
RSOL = ap.constants.R_sun.cgs.value
PC = ap.constants.pc.cgs.value
AU = ap.constants.au.cgs.value
YR = ap.units.year.to(ap.units.s)

MELC = ap.constants.m_e.cgs.value
MPRT = ap.constants.m_p.cgs.value

H0 = cosmo.H0.cgs.value                           # Hubble Constants at z=0.0 
OMEGA_M = cosmo.Om0
OMEGA_B = cosmo.Ob0
OMEGA_DM = cosmo.Odm0
RHO_CRIT = cosmo.critical_density0.cgs.value

'''
HPAR                  = 0.704                               # Hubble parameter little h
PC                    = 3.085678e+18                        # Parsec  in cm
MSOL                  = 1.989e+33                           # Solar Mass   1 M_sol in g
RSOL                  = 6.96e+10                            # Solar Radius 1 R_sol in cm
MPRT                  = 1.673e-24                           # proton mass in g
NWTG                  = 6.673840e-08                        # Newton's Gravitational  Constant
SPLC                  = 2.997925e+10                        # Speed of light [cm/s]
AU                    = 1.495979e+13                        # Astronomical Unit in cm
YR                    = 3.156e+07                           # Year in seconds
H0                    = 2.268546e-18                        # Hubble constant at z=0.0   in [1/s]
'''

# Derived
PIFT                  = 4.0*np.pi/3.0                       # (4.0/3.0)*Pi
SCHW                  = 2*NWTG/(SPLC*SPLC)                  # Schwarzschild Constant (2*G/c^2)
#RHO_CRIT              = 3.0*H0*H0/(4.0*np.pi*NWTG)          # Cosmological Critical Density [g/cm^3
HTAU                  = 1.0/H0                              # Hubble Time - 1/H0 [sec]

#YEAR                  = YR
MYR                   = 1.0e6*YR
GYR                   = 1.0e9*YR

KPC                   = 1000.0*PC
