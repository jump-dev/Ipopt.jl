# Ipopt.jl

**Ipopt.jl** is a [Julia](http://julialang.org/) interface to the [Ipopt](http://www.coin-or.org/Ipopt/documentation/documentation.html) nonlinear solver.

You can install it using the Julia package manager:

```julia
julia> Pkg.add("Ipopt")
```

This will install Ipopt.jl, as well as Ipopt itself. A binary will be downloaded on Windows and
on OSX (via Homebrew), but it will be built from source on Linux. You should make sure you have
the required packages before installing, e.g.

```bash
sudo apt-get install build-essential gfortran pkg-config
```

Full documentation is available [here](http://ipoptjl.readthedocs.org/en/latest/ipopt.html).

Ipopt.jl has testing, and the status of the current build is [![Build Status](https://travis-ci.org/JuliaOpt/Ipopt.jl.png?branch=master)](https://travis-ci.org/JuliaOpt/Ipopt.jl)
