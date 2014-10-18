# Ipopt.jl

[![Build Status](https://travis-ci.org/JuliaOpt/Ipopt.jl.png?branch=master)](https://travis-ci.org/JuliaOpt/Ipopt.jl)
[![Coverage Status](https://img.shields.io/coveralls/JuliaOpt/Ipopt.jl.svg)](https://coveralls.io/r/JuliaOpt/Ipopt.jl)
[![Ipopt](http://pkg.julialang.org/badges/Ipopt_release.svg)](http://pkg.julialang.org/?pkg=Ipopt&ver=release)

**Ipopt.jl** is a [Julia](http://julialang.org/) interface to the [Ipopt](http://www.coin-or.org/Ipopt/documentation/documentation.html) nonlinear solver.

**Installation**: `julia> Pkg.add("Ipopt")`

This will install Ipopt.jl, as well as Ipopt itself. A binary will be downloaded on Windows and
on OSX (via Homebrew), but it will be built from source on Linux. You should make sure you have
the required packages before installing, e.g.

```bash
sudo apt-get install build-essential gfortran pkg-config
```

Full documentation is available [here](http://ipoptjl.readthedocs.org/en/latest/ipopt.html).
