# How to do profiling in Python

## Profilers

### cProfile

cProfile is preinstalled as part of Python's standard library

You can run any python code through cProfile with this command :

`python3 -m cProfile -o <profile_file_name> -m <python code file/command>`

for this project, the most common use will be :

`python3 -m cProfile -o <profile_file_name> -m pytest namosim/tests/e2e/e2e_test.py::TestE2E::<specific_test>`


## Visualization/Analysis

### Snakeviz

https://jiffyclub.github.io/snakeviz/

Snakeviz is a browser based graphical viewer for the output of Python's cProfile module.

You can install it with the following command :

`pip install snakeviz`

And you can use it with the following command :

`snakeviz <profile_file_name>`

This will start a local web server on port 8080 by default (if running in a container, that port may be changed to another on the OS side, VSCode opens the right one when Ctrl+Clicking the link in the terminal)