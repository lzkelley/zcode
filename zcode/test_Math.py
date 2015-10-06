"""
Testing script for the `Math.py` submodule.

Run with:
    $ nosetests -sv test_Math.py

"""

import numpy as np
import matplotlib.pyplot as plt

import Math as zmath
import InOut as zio

PLOT_DIR = "./test_plots/"

PLOT = True
#PLOT = False

def setup_module(module):
    
    # Make sure plot directory exists
    zio.checkPath(PLOT_DIR)

    return


def teardown_module(module):
    return


def test_smooth_1():

    ARR_SIZE = 1000
    AMP = 10.0
    NOISE = 1.4
    SMOOTH_LENGTHS = [1,4,16]

    xx = np.linspace(-np.pi/4.0, 3.0*np.pi, num=ARR_SIZE)
    arrs = [ AMP*np.sin(xx) + NOISE*np.random.uniform(-1.0,1.0,size=len(xx)) 
             for ii in xrange(len(SMOOTH_LENGTHS)) ]

    smArrs = [ zmath.smooth(arr, smlen) 
               for (arr,smlen) in zip(arrs,SMOOTH_LENGTHS) ]

    if( PLOT ):
        # Plot Results
        fig,ax = plt.subplots(figsize=[10,6])
        plt.subplots_adjust(left=0.05, right=0.85)
        lines = []
        names = []

        for ii,(sm,ar,smlen) in enumerate(zip(smArrs, arrs, SMOOTH_LENGTHS)):
            offs = ii*5.0

            l1, = ax.plot(xx, ar+offs, '-', color='0.5', lw=2.0)
            if( ii == 0 ): 
                lines.append(l1)
                names.append("Input")
            ll, = ax.plot(xx, sm+offs, lw=1.0, alpha=0.5)
            lines.append(ll)
            names.append("Smooth %d" % (smlen))

        fig.legend(lines, names, loc='lower right', ncol=1, prop={'size':10},
                   bbox_transform=fig.transFigure, bbox_to_anchor=[0.99, 0.01])

        fname = PLOT_DIR + "test_smooth_1.png"
        fig.savefig(fname)
        print "Saved to '%s'" % (fname)

        
    # Variance between smoothed and raw should be decreasing
    stdDiffs = [ np.std(sm-arr) for sm in smArrs ]
    assert stdDiffs[0] > stdDiffs[1] > stdDiffs[2]

    # Smoothing length 1 should have no effect
    assert np.all( smArrs[0] == arrs[0] )
    
    return
    

def test_smooth_2():

    ARR_SIZE = 1000
    AMP = 10.0
    NOISE = 1.4
    SMOOTH_LENGTHS = [4,16,32]
    WIDTH = [ [0.25,0.75], 100, 100 ]
    LOCS  = [ None, 800, 0.8 ]

    xx = np.linspace(-np.pi/4.0, 3.0*np.pi, num=ARR_SIZE)
    arr = AMP*np.sin(xx) + NOISE*np.random.uniform(-1.0,1.0,size=ARR_SIZE)

    smArrs = [ zmath.smooth(arr, smlen, width=wid, loc=loc) 
               for smlen,wid,loc in zip(SMOOTH_LENGTHS,WIDTH,LOCS) ]


    if( PLOT ):
        # Plot Results
        fig,ax = plt.subplots(figsize=[10,6])
        plt.subplots_adjust(left=0.05, right=0.85)
        lines = []
        names = []
        l1, = ax.plot(xx,arr, '-', color='0.5', lw=2.0)
        lines.append(l1)
        names.append("Input")

        for ii,(sm,smlen,wid,loc) in enumerate(zip(smArrs, SMOOTH_LENGTHS, WIDTH, LOCS)):
            offset = ii*5.0
            ax.plot(xx, arr+offset, '-', color='0.5', lw=2.0)
            ll, = ax.plot(xx, sm+offset, lw=1.0, alpha=0.75)
            lines.append(ll)
            names.append("Smooth %d, at %s %s" % (smlen, str(wid), str(loc)))

        fig.legend(lines, names, loc='lower right', ncol=1, prop={'size':10},
                   bbox_transform=fig.transFigure, bbox_to_anchor=[0.99, 0.01])

        fname = PLOT_DIR + "test_smooth_2.png"
        fig.savefig(fname)
        print "Saved to '%s'" % (fname)


    assert np.all( smArrs[0][:249] == arr[:249] )
    assert np.all( smArrs[0][751:] == arr[751:] )

    assert np.all( smArrs[1][:700] == arr[:700] )
    assert np.all( smArrs[1][901:] == arr[901:] )

    assert np.all( smArrs[2][:699] == arr[:699] )
    assert np.all( smArrs[2][901:] == arr[901:] )

    return
