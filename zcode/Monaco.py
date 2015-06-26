
import numpy as np
import sys
import matplotlib as mpl
from matplotlib import pyplot as plt
import math
import random
import time

# from astropy import io.ascii as at

sigmaData = 0.1          # Standard deviation of each input data point
sigmaA = 0.2            # StdDev in step size for A
sigmaB = 0.2            # StdDev in step size for B

xrange = [-1.0, 1.0]     # X-axis range for plots
yrange = [-0.5, 1.5]     # Y-axis   "    "    "
arange = [0.2, 1.5]      # A (y-intercept) range for plots
brange = [0.0, 1.5]      # B (slope)         "    "    " 

# Initial Guess for line parameters
a0 = 1.2                 # Y-Inercept
b0 = 0.2                 # Slope

# MCMC Parameters
nsteps = 100            # Number of total steps to attempt
burnin = 60             # Number of points to consider 'burn-in'
animateflag = True      # Create animation (slow)
plotflag = True          # Create plots (fast)
nhistbins = 40           # Number of histogram bins per variable



# linear() - Find linear f(x|a,b) = a + b*x 
def linear(x,ta,tb):
    return ta + tb*x

# chiSquared() - Find chi-squared between gien data, and linear fit from linear()
def chiSquared(xs,ys,sig,ta,tb):
    chi2 = 0.0
    for i in range(xs.size): chi2 += ((ys[i] - linear(xs[i],ta,tb))/sig)**2
    return chi2

# likelihood() - Calculate likelihood ratio of chi-squared values
def likelihood(cs1, cs2):
    return math.exp(0.5*cs1 - 0.5*cs2)

# animateMC() - Modify plot elements for each frame
def animateMC(tnum):
    tempa = alist[tnum]
    tempb = blist[tnum]
    frac = 1.0 - 1.0*(tnum+1)/(nsteps)
    xx = xguess
    yy = linear(xx, tempa, tempb)

    # - Subplot 1 : Data and line fit
    # Redraw initial guess for fit
    lineguess, = ax1.plot(xguess, yguess, 'g-')
    # Redraw data points
    inline, = ax1.plot(inx, iny, 'bo', markersize=2)
    # Reset text
    animNum.set_text("y = %.3f + %.3fx : %4d/%d" % (tempa, tempb, (1+tnum), nsteps) )
    # Plot the current fit line
    colorval = scalarMap.to_rgba(frac)
    ax1.plot( xx, yy, '-', color=colorval, lw = 0.7)    

    # - Subplot 2 : Markov-chain parameter space
    if( tnum > 0 ): ax2.plot( [blist[tnum-1],blist[tnum]], [alist[tnum-1],alist[tnum]], '-', color=colorval, lw=0.6)

    # - Subplot 3 : Time series of values
    ax3.plot( [tnum,tnum+1],[ math.log10(chilist[tnum-1]/num),math.log10(chilist[tnum]/num)], 'k-', lw=0.5 )
    ax3twin.plot( [tnum,tnum+1],[alist[tnum-1],alist[tnum]], 'b-', lw=0.5 )
    ax3twin.plot( [tnum,tnum+1],[blist[tnum-1],blist[tnum]], 'g-', lw=0.5 )

    # Print progress
    tools.printProgress(tnum, nsteps, time.time()-animStart )




if __name__ == "__main__":
    

    print "\n - Monaco.py - MCMC"
    print "  ",time.strftime("%Y %m %d - %H:%M:%S\n",  time.localtime(time.time()) )
    startTime = time.time()                                                                                         # Store the start Time

    # - Load input data
    indata = at.read(input_filename, names=[ 'n' , 'x' , 'y' ], data_start=0 )                                      # Load data from file
    inx = indata['x']                                                                                               # store x values
    iny = indata['y']                                                                                               # store y values
    num = max(indata['n'])                                                                                          # get number of data points
    print " - - Retrieved %d data points from file '%s'" % (num, input_filename)


    # ================================================  MCMC  ======================================================
    print "\n - - Beginning MCMC"
    print " - - - StdDev in sampling distribution: sigmaA, sigmaB = %.3e, %.3e" % (sigmaA, sigmaB)

    # Initialize the current location parameters, calculate chi-squared 
    a1 = a0
    b1 = b0
    chiSq1 = chiSquared(inx, iny, sigmaData, a0, b0)
    print " - - - Guess for a0,b0 = %.3e,%.3e; initial ChiSquared/N = %.3e" % (a0,b0, chiSq1)

    # Store lists of a,b,chi values etc 
    alist = [ a1 ]                                              # List of current fit for each step (i.e. new value if accepted, old if not)
    blist = [ b1 ]                                              # "
    chilist = [ chiSq1 ]                                        # "
    asteplist = [ a1 ]                                          # List of all steps attempted (regardless of taken or not)
    bsteplist = [ b1 ]                                          # "
    ayes = [ a1 ]                                               # List of steps taken
    byes = [ b1 ]                                               # " 
    ano = []                                                    # List of steps not-taken
    bno = []                                                    # "
    acceptlist = [ True ]                                       # List of steps accepted or not (True / False)
    likes = []                                                  # List of likelihoood ratios for all steps

    naccept = 0                                                 # Store the number of 'accepted' steps
    accept = False
    # - Start Markov Chain
    for i in range(nsteps):
        # Take a step
        a2 = a1 + random.normalvariate(0.0, sigmaA)
        b2 = b1 + random.normalvariate(0.0, sigmaB)
        asteplist.append( a2 )
        bsteplist.append( b2 )
        chiSq2 = chiSquared(inx, iny, sigmaData, a2, b2)
        # Calculate the likelihood ratio, compare to random variable
        likerat = likelihood(chiSq1, chiSq2)
        likes.append( likerat )
        thresh = random.uniform(0.0,1.0)
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

    print " - - - Accepted %d, Rejected %d = %.2f" % (naccept, nsteps-naccept, 1.0*naccept/nsteps)

    numgood = len(alist[burnin:])
    # Calculate Means
    amean = np.sum(alist[burnin:])/numgood
    bmean = np.sum(blist[burnin:])/numgood
    # Calculate Std Devs
    astdev = math.sqrt( np.sum( (alist[burnin:] - amean)**2 )/numgood )
    bstdev = math.sqrt( np.sum( (blist[burnin:] - bmean)**2 )/numgood )

    print " - - - A = %.3e +- %.3e   B = %.3e +- %.3e" % (amean, astdev, bmean, bstdev)



    # ================================================  Plots  =====================================================
    if( plotflag ):
        print " - - Plotting data"

        # ---- Plot Time Series
        fig1name = plot_data_filename_prefix + "time_series" + plot_data_filename_suffix
        plotfig1 = plt.figure(figsize=[8,5])
        plotax1 = plt.subplot(111)
        plotax1.set_xscale('log')
        plotax1.set_xlim( [1,nsteps+1] )
        plotax1.xaxis.grid(True)
        plotax1.yaxis.grid(True)
        abmin = min( min(alist),min(blist) )
        abmax = max( max(alist),max(blist) )
        chimin = min(chilist/num)
        chimax = max(chilist/num)
        plotax1.set_ylim( [0.9*abmin, 1.1*abmax] )
        plotax1twin = plotax1.twinx()
        plotax1.set_ylabel("a,b")
        plotax1.set_xlabel("Step Number")
        plotax1.set_title("Parameter Time Series")
        plotax1twin.set_ylabel("Chi2/N")
        plotax1twin.set_yscale('log')
        plotax1twin.set_ylim( 0.2*chimin, 2*chimax )
        plotfig1.tight_layout()

        aline, = plotax1.plot(range(1,nsteps+2), alist, 'b-')
        bline, = plotax1.plot(range(1,nsteps+2), blist, 'g-')
        chiline, = plotax1twin.plot(range(1,nsteps+2), chilist/(num-1), 'r-')

        plt.legend([aline, bline, chiline], ["A" , "B", "Chi2/N"])
        plotfig1.savefig(fig1name, dpi=plotDPI)


        # ---- Plot Histograms
        fig2name = plot_data_filename_prefix + "histograms" + plot_data_filename_suffix
        plotfig2 = plt.figure(figsize=[8,5])
        plotax2 = plt.subplot(111)
        plotax2.set_title("Posterior Distribution Function")
        plotfig2.tight_layout()

        ahist = plt.hist(alist[burnin:], nhistbins, normed=True, histtype='bar', alpha=0.5, label="A")
        bhist = plt.hist(blist[burnin:], nhistbins, normed=True, histtype='bar', alpha=0.5, label="B")

        plt.figtext(0.5,0.76,"A = %.3f +- %.3f\nB = %.3f +- %.3f" % (amean, astdev, bmean, bstdev) )

        plt.legend()
        plotfig2.savefig(fig2name, dpi=plotDPI)


        # ---- Plot Data and Fit Line
        fig3name = plot_data_filename_prefix + "fit" + plot_data_filename_suffix
        plotfig3 = plt.figure(figsize=[8,5])
        fig3ax1 = plt.subplot2grid((3,1),(0,0), rowspan=2)
        fig3ax2 = plt.subplot2grid((3,1),(2,0))
        fig3ax1.xaxis.grid(True)
        fig3ax1.yaxis.grid(True)
        fig3ax2.xaxis.grid(True)
        fig3ax2.yaxis.grid(True)
        fig3ax1.set_ylabel("y")
        fig3ax2.set_ylabel("Residuals")
        fig3ax2.set_xlabel("x")

        plotfig3.tight_layout()

        # Calculate residuals
        resdat = np.zeros(num)
        for i in range(num):
            resdat[i] = iny[i] - linear(inx[i],amean,bmean)

        xguess = np.array(xrange)
        yguess = linear(xguess, a0, b0)
        guessline, = fig3ax1.plot(xguess, yguess, 'g-', lw=1.5)
        dpoints, = fig3ax1.plot(inx, iny, 'bo', markersize=6)
        yfit = linear(xguess, amean, bmean)
        fitline, = fig3ax1.plot( xguess, yfit, 'r-', lw = 2.0)
        resline, = fig3ax2.plot( inx, resdat, 'ro', markersize=6)

        fig3ax1.legend( [dpoints, guessline, fitline] , ["Data", "Guess (y=%.2f+%.2fx)" % (a0,b0), "Fit (y=%.3f+%.3fx)" % (amean, bmean)], loc=2)
        # plt.figtext(0.5,0.76,"A = %.3f +- %.3f\nB = %.3f +- %.3f" % (amean, astdev, bmean, bstdev) )

        plotfig3.savefig(fig3name, dpi=plotDPI)



        # ---- Plot Markov-Chain in Parameter Space
        fig4name = plot_data_filename_prefix + "markov_chain" + plot_data_filename_suffix
        plotfig4 = plt.figure(figsize=[8,5])
        plotax4 = plt.subplot(111)
        plotax4.xaxis.grid(True)
        plotax4.yaxis.grid(True)
        plotax4.set_ylabel("A")
        plotax4.set_xlabel("B")
        plotax4.set_title("Markov Chain")
        plotfig4.tight_layout()

        noline, = plotax4.plot(bno,ano, '-', color='grey', lw=0.6)
        yesline2, = plotax4.plot(byes,ayes,'k-', lw=1.4)
        yesline, = plotax4.plot(byes,ayes,'r-', lw=1.0)
        meandot, = plotax4.plot(bmean,amean, 'bo', markersize=3)
        e1 = mpl.patches.Ellipse((bmean, amean), 2*bstdev, 2*astdev, linewidth=2, fill=True, alpha=0.6, color='blue', zorder=10)
        plotax4.add_patch(e1)

        #plt.legend([aline, bline, chiline], ["A" , "B", "Chi2"])
        plotfig4.savefig(fig4name, dpi=plotDPI)











    # =================================================  Animation  =================================================
    if( animateflag ):
        print " - - Animating Data"
        # - Create Figure for input data 
        print " - - - Creating figure"
        animfig = plt.figure(figsize=[5,3.5])                                               # Create a figure
        ax1 = plt.subplot2grid((2,2),(0,0), colspan=2)                                      # Axes for data/fit-line
        ax2 = plt.subplot2grid((2,2),(1,0))                                                 # Axes for parameter space
        ax3 = plt.subplot2grid((2,2),(1,1))                                                 # Axes for time-series params and StdDevs
        # Set axes properties
        ax1.set_xlim(xrange)
        ax1.set_ylim(yrange)
        ax1.set_ylabel('y')
        ax1.set_xlabel('x')
        ax1.xaxis.grid(True)
        ax1.yaxis.grid(True)
        ax2.set_xlim(brange)
        ax2.set_ylim(arange)
        ax2.set_xlabel('b')
        ax2.set_ylabel('a')
        ax2.xaxis.grid(True)
        ax2.yaxis.grid(True)
        ax3.set_xlabel('steps')
        ax3.set_ylabel('Log(ChiSquared/N)')
        ax3.xaxis.grid(True)
        ax3.yaxis.grid(True)
        # set figure properties
        plt.tight_layout()
        mpl.rcParams.update({'font.size': 5})
        # Define a colormap
        colmap = plt.get_cmap('winter')
        cNorm  = mpl.colors.Normalize(vmin=0.0, vmax=1.0) #(vmin=0, vmax=naccept)            # normalize colormap to [0.0,1.0]
        scalarMap = mpl.cm.ScalarMappable(norm=cNorm, cmap=colmap)                           # Convert to a scalar map

        # Construct elements to appear in figure
        figTitle = plt.figtext(0.15,0.94,"Linear Fit")                                       # Manually cosntruct title
        animNum = plt.figtext(0.60,0.94,"")                                                  # Write which frame we're in 
        # Create line for initial guess
        xguess = np.array(xrange)
        yguess = linear(xguess, a0, b0)                                                      # Calculate y-values of initial guess
        # Create line for the current fit
        linefit, = ax1.plot( [], [], 'r-')
        # Create line for a-b parameter space
        parspace, = ax2.plot( [], [], 'k-')
        # Create dot for starting point in parameter space
        ax2.plot( b0, a0, 'ko', markersize=2)

        ax3.set_xlim([0,nsteps])
        ax3.set_ylim([ math.log10(0.9*min(chilist)/num) , math.log10(1.1*max(chilist)/num) ])
        ax3twin = ax3.twinx()
        ax3twin.set_ylim( [min([arange[0],brange[0]]), max([arange[1],brange[1]]) ] )
        ax3twin.set_ylabel('a,b,sigma')

        # - Create animation
        animStart = time.time()
        print " - - - Creating animation"
        animPars = anim.FuncAnimation(animfig, animateMC, frames=nsteps, blit=True)
        print " - - - Saving to '%s'" % anim_data_filename
        animPars.save(anim_data_filename, extra_args=['-vcodec', 'libx264'], dpi=animDPI, bitrate=numBR, fps=numFPS)




    endTime = time.time()
    print "\n - Done after", tools.stringTime(endTime-startTime),"\n"










    # ---------------------------------------------------
    # -----------------------------------------------------------------------------------------------
