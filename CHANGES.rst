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
        -   `checkPath`
            +   [ENH] added parameter `create` to choose whether missing directories are created
                or not.
            +   [DOC] added docstrings.
        -   `ascii_table`
            +   [ENH] passing ``out = None`` will make the function return a string version of the
                table.
    +   timer.py
        -   [ENH] `Timings.report()` will return the results as a string if the parameter,
            ``out = None``.
-   math/
    +   math_core.py
        -   `_comparisonFunction` [DEPRECATED] ---> `_comparison_function` [new-function]
            +   [ENH] Returned function takes a single parameter, instead of needing the comparison
                value in each call.  Instead the comparison value is passed once to
                `_comparison_function`, just during initialization.
        -   `_comparisonFilter` [DEPRECATED] ---> `comparison_filter` [new-function]
            +   [ENH] Added options to return indices (instead of values), compare with non-zero
                comparison values, and check for finite (or not).
        -   `stats_str`
            +   [ENH] Added parameter `label` to prepend to the output string.
    +   tests/
        -   test_math_core.py
            +   [ENH] Added *some* tests for `_comparison_function` and `_comparison_filter`.
-   plot/
    +   Hist2D.py
        -   `plot2DHistProj`
            +   [ENH] added parameters to adjust the size / location of axes composing plots.
            +   [BUG] fixed issue where log-color-scales projected axes with zero values would
                fail.  Seems to be working fine.


[0.0.6] - 2016/01/30
--------------------
-   constants.py
    +   Bug-fix where `SIGMA_T` wasn't loading properly from `astropy`.
    +   Added Electron-Scattering opacity, `KAPPA_ES`.
-   README.rst
    +   Added more information about contents and structure of package.
-   inout/
    +   inout_core.py
        -   `ascii_table` [new-function]
            +   New function which prints a table of values to the given output.
            +   Added `linewise` and `prepend` arguments, allowing the table to be printed
                line-by-line or as a single block, and for the print to be prepended with
                an additional string.
        -   `modify_exists` [new-function]
            +   Function which modifies the given filename if it already exists.  The modifications
                is appending an integer to the filename.
            +   Added tests for this function.
    +   timer.py [new-file]
        -   Provides the classes `Timer` and `Timings` which are used to time code execution and
            provided summaries of the results.  The `Timer` class is used to calculate repeated
            durations of execution for the same (type of) calculation, while the `Timings` class
            will manage the timing of many different calculations/chunks of code.
    +   tests/
        -   test_inout_core.py
            +   Fixed some issues with cleaning up (deleting) files/directories created for the
                tests.
        -   test_timer.py [new-file]
            +   Test for the classes in the new `inout/timer.py` file.  Basics tests in place.

-   math/
    +   math_core.py
        -   `groupDigitized`
            +   [Docs]: improved documentation clarifying input parameters.
        -   `stats_str` [new-function]
            +   [ENH]: Return a string with the statistics of the given array.
        -   `_comparisonFilter`
            +   [ENH]: always filter for finite values (regardless of the function arguments).
-   plot/
    +   plot_core.py
        -   `plotConfFill`
            +   [Bug]: fixed default value of `outline` which was still set to a boolean instead of
                a color string.  Caused failure when trying to save images.
        -   `colorCycle` [DEPRECATED] ---> `color_cycle` [new-function]
            +   [Docs]: added method documentation.
    +   Hist2D.py
        -   `plot2DHistProj`
            +   [ENH]: Check to make input arguments are the correct (consistent) shapes.
            +   [ENH]: Added flag 'write_counts' which overlays a string of the number of values in
                each bin of the 2D histogram.  Uses the new `counts` parameter of `plot2DHist`.
        -   `plot2DHist`
            +   [ENH]: Added parameter 'counts' for numbers to be overlaid on each bin, used by
                the `write_counts` of `plot2DHistProj`.


[0.0.5] - 2015/12/13
--------------------
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
            +   Added an `outline` argument to optional draw a line with a different color
                behind the median line, to make it more visible.
        -   `text`
            +   [Bug]: fixed issue where regardless of what transformation was passed, only the
                `figure` transformation was used.  Solution is to call ``plt.text`` instead of
                ``fig.text``.
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
