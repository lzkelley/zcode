zcode
=====
Python package for general purpose utility code for a variety of (scientific/computational)
applications.

Package Organization
-----------------------
### Contents:
-   zcode/                        - Actual source code.
    +   inout/                    - IO.
        -   inout_core.py         - Basics.
        -   log.py                - Logging related.
    +   math/                     - Math and array.
        -   hist.py               - Histograms and binning.
        -   math_core.py          - Basics.
    +   plot/                     - Plotting and visualization.
        -   color2d.py            - Creating 2D colormaps.
        -   CorrelationGrid.py    - Triangle plot of parameter correlations and histograms.
        -   Hist2D.py             - Creating 2D histograms with projections.
        -   plot_core.py          - Basics.
    +   test.sh                   - Simple script to run `nosetests`.
-   CHANGES.rst                   - Versioned updates to the code base.
-   LICENSE                       - Standard license information.
-   MANIFEST.in                   - File inventory for packaging information.
-   setup.py                      - Python setup script for installation.
-   setup.sh                      - Standard installation for package.
-   version.py                    - Version file for setup-script (superfluous?).

#### Tests:
Each subdirectory (e.g. 'math/') contains a 'tests/' directory which includes methods for internal
diagnostics and testing.  These can all be run using the `zcode/test.sh` script.

#### Deprecations:
When deprecating a function/object/method, use a warning like:
``warnings.warn(warnStr.format('positive'), DeprecationWarning, stacklevel=3)``
