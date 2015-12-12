CHANGES
=======

Future / To-Do
--------------
-   General
    +   Implement tests for all modules and functions.
-   math/
    +   math_core.py
        -   `spacing`
            +   Add `endpoint` keyword argument to decide whether end-point(s) are included
                in returned range.


Current
-------
-   inout/
    +   inout_core.py
        -   `dictToNPZ`
            +   Added optional `log` parameter for a ``logging.Logger`` object.
            +   Instead of raising an error for scalar parameters, cast them into arrays and
                print a warning.
    +   tests/
        -   `test_inout_core.py` [new-file]
            +   Tests for the `inout_core.py` submodule.
            +   Added tests for `npzToDict` and `dictToNPZ`.
-   math/
    +   math_core.py
        -   `confidenceBands`
            +   Added `filter` argument to select points based on how their `y` values compare to
                zero, e.g. to select for only ``y >= 0.0`` etc.
        -   `minmax`
            +   Added a `filter` argument to replace usage of `nonzero` (use `'!='`) and
                `positive` (use `'>'`).  Left both of the arguments in place, but usage of them
                will print a deprecation warning.
        -   `spacing`
            +   Updated to use `filter` argument.
-   plot/
    +   plot_core.py
        -   `plotConfFill`
            +   Added a `filter` argument to filter the values to be plotted.
    +   color2d.py [new-file]
        -   New file with classes and functions to provide color-mappings from 2D parameter spaces
            to RGB color-space.  `ScalarMappable2D` is the class which handles this mapping,
            analogous to the `matplotlib.cm.ScalarMappable` class.  Similarly, the function to
            create an instance is `zplot.color2d.colormap2d`, analogous to the
            `zcode.plot.plot_core.colormap` function.
-   constants.py
    +   Added `sigma_T` -- the Thomson-scattering cross-section in units of cm^2.


[0.0.4] - 2015/11/19
--------------------
-   General
    +   Can now run tests through python via ``>>> zcode.test()``.
-   inout/
    +   inout_core.py
        -   `mpiError` [new-method]
            +   New method to raise an error across an MPI communicator
    +   log.py
        -   `getLogger`
            +   Added the log output filename as a member variable to the newly created
                logger object.
-   math/
    +   math_core.py
        -   `argextrema` [new-method]
            +   Method to find the index of the extrema (either 'min' or 'max') with filtering
                criteria (e.g. 'ge' = filter for values ``>= 0.0``).
        -   `really1d` [new-method]
            +   Test whether a list or array is purely 1D, i.e. make sure it is not a 'jagged'
                list (or array) of lists (or arrays).
        -   `asBinEdges` [new-method]
            +   Convert a bin-specification to a list of bin-edges.  I.e. given either a set of
                bin-edges, or a number of bins (in N-dimensions), return or create those bin-edges.
        -   `confidenceIntervals` [new-method]
            +   For a pair of x and y data, bin the values by x to construct confidence intervals
                in y.
    +   tests/
        -   test_math_core.py [new-file]
            +   New location and standard for math tests using 'nose'.
            +   Moved over one of the tests for 'smooth' from previous location,
                'zcode/testing/test_Math.py' [deleted], and simplified.
-   test.sh [new-file]
    +   Bash script containing the single command to use for running nosetests.
-   testing/ [Deleted]
    +   Moved and reformatted test into new 'zcode/math/tests/test_math_core.py' file.


[0.0.3] - 2015/11/09
--------------------
-   Overall
    +   Restructured module to use subdirectories per topic (e.g. 'math') instead of single files.
    +   Implemented python3 styles into all files, with backwards compatibility.
-   CHANGES.rst [new-file]
    +   Track changes.
-   MANIFEST.in [new-file]
    +   Track files required for module.
-   version.py  [new-file]
    +   Current version information loaded from 'zcode.__init__'.
    +   Should be expanded to include git commit SHA, etc.
-   math/
    +   math_core.py
        -   Enhanced the `spline` function, and removed the secondary functions `logSpline` and
            `logSpline_resample`.  The former is included in the new functionality of `spline`,
            and the latter is too simple to warrant its own function.
        -   `strArray [new-function]
            +   Creates a string representation of a numerical array.
        -   `indsWIthin` [new-function]
            +   Finds the indices of an array within the bounds of the given extrema.
        -   `midpoints`
            +   Enhanced to find the midpoints along an arbitrary axis.
-   plot/
    +   plot_core.py
        -   `legend` [new-method]
            +   Similar to 'text' --- just a wrapper for `matplotlib.pyplot.legend`.
        -   `plotConfFill` [new-method]
            +    Draws a median line and filled-regions for associated confidence intervals
                 (e.g. generated by `zcode.math.confidenceIntervals`).
    +   Hist2D.py
        -   Plotted histograms now use the `scipy.stats.binned_statistic` function so that more
            complicated statistics can be used.  The projected histograms are now colored to match
            the 2D main histogram.
-   inout/
    +   inout_core.py
        -   `MPI_TAGS` [new-class]
            +    A `Keys` subclass used for passing tags/status between different processors when
                 using MPI.  Commonly used in the master-slave(s) paradigm.

[0.0.2] - 2015/10/20
--------------------
