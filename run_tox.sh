#!/bin/zsh
#
# `pyenv` can be installed with homebrew, e.g. `brew install pyenv`
# NOTE: the versions of python (in `VERSIONS`) need to match those in the `tox.ini` file's `envlist` variable
#

set -e
VERSIONS=("3.9:latest" "3.10:latest")

# Iterate the string array using for loop
for val in ${VERSIONS[@]}; do
   echo $val
   pyenv install -s $val
done

# versions=("${(@f)$(pyenv versions)}")
# versions=("${versions[@]:1}")
# echo ${versions}
# clean=()
# for val in ${versions[@]}; do
#    val=${val// }
#    clean+=($val)
# done
# echo $clean
# pyenv local ${clean[@]}

tox