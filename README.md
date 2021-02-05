This repository is an ammeded version of D.Hanson's [Quicklens code](https://github.com/dhanson/quicklens). It includes the fixes listed below, and a branch translating it to Python 3 (in progress).

# Fixed bugs
### 1) Spherical harmonic transforms for fields with non-zero spin

Faulty functions: shts/\_\_init\_\_.map2vlm() and shts/\_\_init\_\_.vlm2map()

What we were seeing was that, while the TT full-sky reconstruction worked fine, things went wrong pretty dramatically as soon as you tried to incorporate polarisation fields. We seem to have tracked this down to a bug in the functions that convert between real space fields and their spin-s spherical harmonic representation. In particular, it appears that the issue had to do with how {s}\_Y\_{lm} was manipulated when s<0 to avoid feeding the fortran module actually doing the transformation a negative spin. The bug was harmless when s=0, so temperature worked fine. 

### 2) Simulation of TEB fields in the flat-sky limit

Faulty function: sims/sims.tebfft()

The bug would have appeared if you ever used it to simulate flat-sky fields: the function ql.sims.tebfft() can be used to generate realizations of size (nx,ny) of real TEB fields by drawing complex Fourier coefficients with the right power spectrum on a Fourier-space grid of size (nx, ny/2+1). However, the implementation forgets to set the strip (lx=0,ly<0) to the complex conjugate of the (lx=0, ly>0) entries. One manifestation of this bug is that the lensed simulations you generate will lack power, with the shortage being most acute on large scales. 

### 3) Multithreading bug
Faulty function: shts/shts.f95

Threads were getting in each others' way when more than one was used.
# Added curved-sky functions

Location: sims/ivf.py and sims/qest.py.

Added ivf.library_diag_full_sky(), qest.get_qft_full_sky() and qest.get_qr_full_sky() which enable inverse-variance filtering and lensing reconstruction on the curved-sky based on the quadratic estimators implemented by Hanson.
