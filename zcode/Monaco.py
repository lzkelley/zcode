"""
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from matplotlib import pyplot as plt

import zcode.Plotting as zplot
import zcode.Math     as zmath

# from astropy import io.ascii as at

#sigmaData = 0.1          # Standard deviation of each input data point
sigmaA = 0.1            # StdDev in step size for A
sigmaB = 0.1            # StdDev in step size for B


xrange = [-1.0, 1.0]     # X-axis range for plots
yrange = [-0.5, 1.5]     # Y-axis   "    "    "
arange = [0.2, 1.5]      # A (y-intercept) range for plots
brange = [0.0, 1.5]      # B (slope)         "    "    "

# Initial Guess for line parameters
a0 = 0.5                 # Y-Inercept
b0 = 4.0                 # Slope

# MCMC Parameters
nsteps = 1000            # Number of total steps to attempt
burnin = 100             # Number of points to consider 'burn-in'
#animateflag = True      # Create animation (slow)
#plotflag = True          # Create plots (fast)
#nhistbins = 40           # Number of histogram bins per variable






# linear() - Find linear f(x|a,b) = a + b*x
def linear(x,ta,tb):
    return ta + tb*x

# chiSquared() - Find chi-squared between given data, and linear fit from linear()
def chiSquared(xs,ys,sig,ta,tb):
    chi2 = 0.0
    for i in range(xs.size): chi2 += ((ys[i] - linear(xs[i],ta,tb))/sig)**2
    return chi2

# likelihood() - Calculate likelihood ratio of chi-squared values
def likelihood(cs1, cs2):
    return np.exp(0.5*cs1 - 0.5*cs2)







def main():

    print("\n - Monaco.py - MCMC")

    num = 100
    SIGMA_X = 0.01
    SIGMA_Y = 0.4
    AMP     = 10.0
    inx = np.linspace(0.0, 1.0, num=num) + np.random.uniform(-SIGMA_X, SIGMA_X, size=num)
    iny = AMP*(inx + np.random.uniform(-SIGMA_Y, SIGMA_Y, size=num))

    plt.scatter(inx, iny)
    # plt.savefig('test.png')


    # sigmaData = np.std(iny)
    sigmaData = 1.0
    #sigmaA = 0.01
    #sigmaB = 0.01



    # ================================================  MCMC  ====================================
    print("\n - - Beginning MCMC")
    print(" - - - StdDev in sampling distribution: sigmaA, sigmaB = %.3e, %.3e" % (sigmaA, sigmaB))

    # Initialize the current location parameters, calculate chi-squared
    a1 = a0
    b1 = b0
    chiSq1 = chiSquared(inx, iny, sigmaData, a0, b0)
    print(" - - - Guess for a0,b0 = %.3e,%.3e; initial ChiSquared/N = %.3e" % (a0,b0, chiSq1))

    # Store lists of a,b,chi values etc
    alist = [ a1 ]      # List of current fit for each step (i.e. new value if accepted, old if not)
    blist = [ b1 ]
    chilist = [ chiSq1 ]
    asteplist = [ a1 ]             # List of all steps attempted (regardless of taken or not)
    bsteplist = [ b1 ]
    ayes = [ a1 ]                  # List of steps taken
    byes = [ b1 ]
    ano = []                        # List of steps not-taken
    bno = []
    acceptlist = [ True ]               # List of steps accepted or not (True / False)
    likes = []                           # List of likelihoood ratios for all steps

    naccept = 0                                 # Store the number of 'accepted' steps
    accept = False
    # - Start Markov Chain
    for i in range(nsteps):
        # Take a step
        a2 = a1 + np.random.normal(0.0, sigmaA)
        b2 = b1 + np.random.normal(0.0, sigmaB)
        asteplist.append( a2 )
        bsteplist.append( b2 )
        chiSq2 = chiSquared(inx, iny, sigmaData, a2, b2)
        # Calculate the likelihood ratio, compare to random variable
        likerat = likelihood(chiSq1, chiSq2)
        likes.append( likerat )
        thresh = np.random.uniform(0.0,1.0)
        # Accept step if meets criteria
        if( likerat > thresh ):
            accept = True
            a1 = a2
            b1 = b2
            chiSq1 = chiSq2
            ayes.append( a1 )
            byes.append( b1 )
            naccept += 1
        else:
            ano.append( a2 )          # Store failed step
            ano.append( a1 )
            bno.append( b2 )          # Store saved point, for plotting purposes
            bno.append( b1 )

        acceptlist.append( accept )
        # Store new positions (regardless of excepted or not)
        alist.append( a1 )
        blist.append( b1 )
        chilist.append( chiSq1 )

    print(" - - - Accepted %d, Rejected %d = %.2f" % (naccept, nsteps-naccept, 1.0*naccept/nsteps))
    print(" Final chi2 = %e" % (chilist[-1]))

    numgood = len(alist[burnin:])
    # Calculate Means
    amean = np.sum(alist[burnin:])/numgood
    bmean = np.sum(blist[burnin:])/numgood
    # Calculate Std Devs
    astdev = np.sqrt( np.sum( (alist[burnin:] - amean)**2 )/numgood )
    bstdev = np.sqrt( np.sum( (blist[burnin:] - bmean)**2 )/numgood )

    print(" - - - A = %.3e +- %.3e   B = %.3e +- %.3e" % (amean, astdev, bmean, bstdev))


    outx = np.linspace(0.0, 1.0, num=100)
    # outy = amean*outx + bmean
    outy = bmean*outx + amean
    plt.plot(outx, outy, 'k-')
    plt.savefig('test.png')

    plt.clf()
    n, bins, patches = plt.hist(alist[burnin:], 100) #, histtype='stepfilled')
    plt.savefig('a.png')

    plt.clf()
    n, bins, patches = plt.hist(blist[burnin:], 100) #, histtype='stepfilled')
    plt.savefig('b.png')

    plt.clf()
    ax = plt.gca()
    zplot.plotSegmentedLine(ax, alist, blist)
    ax.set_xlabel('a'); ax.set_ylabel('b')
    ax.set_xlim(zmath.minmax(alist))
    ax.set_ylim(zmath.minmax(blist))
    plt.savefig('hist.png')

    plt.clf()
    plt.plot(chilist, 'k-')
    plt.savefig('chi.png')

    return

# main()

if __name__ == "__main__": main()
