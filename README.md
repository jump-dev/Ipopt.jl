Ipopt.jl
========

[![Build Status](https://travis-ci.org/JuliaOpt/Ipopt.jl.svg?branch=master)](https://travis-ci.org/JuliaOpt/Ipopt.jl)
[![Coverage Status](https://img.shields.io/coveralls/JuliaOpt/Ipopt.jl.svg)](https://coveralls.io/r/JuliaOpt/Ipopt.jl)

[![Ipopt](http://pkg.julialang.org/badges/Ipopt_0.3.svg)](http://pkg.julialang.org/?pkg=Ipopt&ver=0.3)
[![Ipopt](http://pkg.julialang.org/badges/Ipopt_0.4.svg)](http://pkg.julialang.org/?pkg=Ipopt&ver=0.4)

**Ipopt.jl** is a [Julia](http://julialang.org/) interface to the [Ipopt](http://www.coin-or.org/Ipopt/documentation/documentation.html) nonlinear solver.

**Installation**: `julia> Pkg.add("Ipopt")`

This will install Ipopt.jl, as well as Ipopt itself. A binary will be downloaded on Windows and
on OSX (via Homebrew), but it will be built from source on Linux. You should make sure you have
the required packages before installing, e.g.

```bash
sudo apt-get install build-essential gfortran pkg-config
```

MathProgBase Interface
----------------------

Ipopt implements the solver-independent [MathProgBase](https://github.com/JuliaOpt/MathProgBase.jl) interface,
and so can be used within modeling software like [JuMP](https://github.com/JuliaOpt/JuMP.jl).
The solver object is called ``IpoptSolver``. All options listed in the [Ipopt documentation](http://www.coin-or.org/Ipopt/documentation/node35.html) may be passed directly. For example, you can suppress output by saying ``IpoptSolver(print_level=0)``. If you wish to pass an option specifically for the restoration phase, instead of using the prefix ``resto.``, use the prefix ``resto_``. For example ``IpoptSolver(resto_max_iter=0)``.

C Interface Wrapper
-------------------

Full documentation for the Ipopt C wrapper is available [here](http://ipoptjl.readthedocs.org/en/latest/ipopt.html). Use of the [nonlinear MathProgBase interface](http://mathprogbasejl.readthedocs.org/en/latest/nlp.html) is recommended over the low-level C interface because it permits one to easily switch between solvers.

