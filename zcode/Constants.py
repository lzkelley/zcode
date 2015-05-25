"""
Common Numerical and Physical Constants.

"""

import numpy as np


HPAR                  = 0.704                               # Hubble parameter little h
PC                    = 3.085678e+18                        # Parsec  in cm
MSOL                  = 1.989e+33                           # Solar Mass   1 M_sol in g
RSOL                  = 6.96e+10                            # Solar Radius 1 R_sol in cm
MPRT                  = 1.673e-24                           # proton mass in g
NWTG                  = 6.673840e-08                        # Newton's Gravitational  Constant
YR                    = 3.156e+07                           # Year in seconds
SPLC                  = 2.997925e+10                        # Speed of light [cm/s]
H0                    = 2.268546e-18                        # Hubble constant at z=0.0   in [1/s]
AU                    = 1.495979e+13                        # Astronomical Unit in cm

# Derived
PIFT                  = 4.0*np.pi/3.0                       # (4.0/3.0)*Pi
SCHW                  = 2*NWTG/(SPLC*SPLC)                  # Schwarzschild Constant (2*G/c^2)
RHO_CRIT              = 3.0*H0*H0/(4.0*np.pi*NWTG)          # Cosmological Critical Density [g/cm^3
HTAU                  = 1.0/H0                              # Hubble Time - 1/H0 [sec]

YEAR                  = YR
MYR                   = 1.0e6*YR
GYR                   = 1.0e9*YR

KPC                   = 1000.0*PC
