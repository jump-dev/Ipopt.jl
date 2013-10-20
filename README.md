# Ipopt.jl

**Ipopt.jl** is a [Julia](http://julialang.org/) interface to the [Ipopt](http://www.coin-or.org/Ipopt/documentation/documentation.html) nonlinear solver.

You can install it using the Julia package manager:

```julia
julia> Pkg.add("Ipopt")
```

This will install Ipopt.jl, as well as Ipopt itself (by building from source on Linux, and by downloading the binary on Windows and OSX [via Homebrew]).

Full documentation is available [here](http://ipoptjl.readthedocs.org/en/latest/ipopt.html).

Ipopt.jl has testing, and the status of the current build is [![Build Status](https://travis-ci.org/JuliaOpt/Ipopt.jl.png?branch=master)](https://travis-ci.org/JuliaOpt/Ipopt.jl)
