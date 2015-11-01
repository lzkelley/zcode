CHANGES
=======

-   Overall
    +    Restructured module to use subdirectories per topic (e.g. 'math') instead of single files.
    +    Implemented python3 styles into all files, with backwards compatibility.
-   CHANGES.rst <== newfile
    +    Track changes.
-   MANIFEST.in <== newfile
    +    Track files required for module.
-   version.py  <== newfile
    +    Current version information.  Should be expanded to include git commit SHA, etc.
-   math
    +    math_core
         -   Enhanced the `spline` function, and removed the secondary functions `logSpline` and
             `logSpline_resample`.  The former is included in the new functionality of `spline`,
             and the latter is too simple to warrant its own function.
         -   new 'strArray' function which creates a string representation of a numerical array.

[0.0.2] - 2015/10/20
