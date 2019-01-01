Ipopt.jl
========

[![Build Status](https://travis-ci.org/JuliaOpt/Ipopt.jl.svg?branch=master)](https://travis-ci.org/JuliaOpt/Ipopt.jl)
[![Coverage Status](https://img.shields.io/coveralls/JuliaOpt/Ipopt.jl.svg)](https://coveralls.io/r/JuliaOpt/Ipopt.jl)


**Ipopt.jl** is a [Julia](http://julialang.org/) interface to the [Ipopt](http://www.coin-or.org/Ipopt/documentation/documentation.html) nonlinear solver.

## Installation

The package is registered in `METADATA.jl` and so can be installed with `Pkg.add`.

```
julia> import Pkg; Pkg.add("Ipopt")
```

Ipopt.jl will use [BinaryProvider.jl](https://github.com/JuliaPackaging/BinaryProvider.jl) to automatically install the Ipopt binaries. This should work for both the official Julia binaries from `https://julialang.org/downloads/` and source-builds.

## Custom Installation

To install custom built Ipopt binaries set the environmental variables `JULIA_IPOPT_LIBRARY_PATH` and `JULIA_IPOPT_EXECUTABLE_PATH`, and call `import Pkg; Pkg.build("Ipopt")`. For instance, if the libraries are installed in `/opt/lib` and the executable is in `/opt/bin` just call
```julia
ENV["JULIA_IPOPT_LIBRARY_PATH"] = "/opt/lib"
ENV["JULIA_IPOPT_EXECUTABLE_PATH"] = "/opt/bin"
import Pkg; Pkg.build("Ipopt")
```
If you do not want BinaryProvider to download the default binaries on install set  `JULIA_IPOPT_LIBRARY_PATH` and `JULIA_IPOPT_EXECUTABLE_PATH`  before calling `import Pkg; Pkg.add("Ipopt")`.

To switch back to the default binaries clear `JULIA_IPOPT_LIBRARY_PATH` and `JULIA_IPOPT_EXECUTABLE_PATH`, and call `import Pkg; Pkg.build("Ipopt")`.

MathProgBase Interface
----------------------

Ipopt implements the solver-independent [MathProgBase](https://github.com/JuliaOpt/MathProgBase.jl) interface,
and so can be used within modeling software like [JuMP](https://github.com/JuliaOpt/JuMP.jl).
The solver object is called ``IpoptSolver``. All options listed in the [Ipopt documentation](http://www.coin-or.org/Ipopt/documentation/node40.html) may be passed directly. For example, you can suppress output by saying ``IpoptSolver(print_level=0)``. If you wish to pass an option specifically for the restoration phase, instead of using the prefix ``resto.``, use the prefix ``resto_``. For example ``IpoptSolver(resto_max_iter=0)``.

C Interface Wrapper
-------------------

Full documentation for the Ipopt C wrapper is available [here](http://ipoptjl.readthedocs.org/en/latest/ipopt.html). Use of the [nonlinear MathProgBase interface](http://mathprogbasejl.readthedocs.org/en/latest/nlp.html) is recommended over the low-level C interface because it permits one to easily switch between solvers.
