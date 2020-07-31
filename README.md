Ipopt.jl
========

[![Build Status](https://travis-ci.org/jump-dev/Ipopt.jl.svg?branch=master)](https://travis-ci.org/jump-dev/Ipopt.jl)
[![Coverage Status](https://img.shields.io/coveralls/jump-dev/Ipopt.jl.svg)](https://coveralls.io/r/jump-dev/Ipopt.jl)


**Ipopt.jl** is a [Julia](http://julialang.org/) interface to the [Ipopt](http://www.coin-or.org/Ipopt/documentation/documentation.html) nonlinear solver.

## Installation

The package is registered in `METADATA.jl` and so can be installed with `Pkg.add`.

```
julia> import Pkg; Pkg.add("Ipopt")
```

In addition to installing the `Ipopt.jl` package, this will also download and install the Ipopt binaries. 
(You do _not_ need to install Ipopt separately.)
If you require a custom build of Ipopt, see the instructions below.

## Custom Installation

To install custom built Ipopt binaries set the environmental variables `JULIA_IPOPT_LIBRARY_PATH` and `JULIA_IPOPT_EXECUTABLE_PATH`, and call `import Pkg; Pkg.build("Ipopt")`. For instance, if the libraries are installed in `/opt/lib` and the executable is in `/opt/bin` just call
```julia
ENV["JULIA_IPOPT_LIBRARY_PATH"] = "/opt/lib"
ENV["JULIA_IPOPT_EXECUTABLE_PATH"] = "/opt/bin"
import Pkg; Pkg.build("Ipopt")
```
If you do not want BinaryProvider to download the default binaries on install set  `JULIA_IPOPT_LIBRARY_PATH` and `JULIA_IPOPT_EXECUTABLE_PATH`  before calling `import Pkg; Pkg.add("Ipopt")`.

To switch back to the default binaries clear `JULIA_IPOPT_LIBRARY_PATH` and `JULIA_IPOPT_EXECUTABLE_PATH`, and call `import Pkg; Pkg.build("Ipopt")`.

JuMP and MathOptInterface
----------------------

Ipopt implements the solver-independent [MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl) interface,
and so can be used within modeling software like [JuMP](https://github.com/jump-dev/JuMP.jl).
The solver object is called `Ipopt.Optimizer`. All options listed in the [Ipopt documentation](https://coin-or.github.io/Ipopt/OPTIONS.html#OPTIONS_REF) may be passed directly. For example, you can suppress output by saying `Ipopt.Optimizer(print_level=0)`. If you wish to pass an option specifically for the restoration phase, instead of using the prefix ``resto.``, use the prefix ``resto_``. For example `Ipopt.Optimizer(resto_max_iter=0)`.

You can use Ipopt with JuMP as follows:
```julia
using JuMP, Ipopt
model = Model(with_optimizer(Ipopt.Optimizer, max_cpu_time=60.0))
```

C Interface Wrapper
-------------------

Full documentation for the Ipopt C wrapper is available [here](http://ipoptjl.readthedocs.org/en/latest/ipopt.html). Use of the [nonlinear MathOptInterface interface](https://github.com/jump-dev/MathOptInterface.jl) is recommended over the low-level C interface because it permits one to easily switch between solvers.

## `INVALID_MODEL` error

If you get a termination status `MOI.INVALID_MODEL`, it is probably because you have some undefined value
in your model, e.g., a division by zero. Fix this by removing the division, or by imposing variable bounds
so that you cut of the undefined region.

Instead of
```julia
model = Model(Ipopt.Optimizer)
@variable(model, x)
@NLobjective(model, 1 / x)
```
do
```julia
model = Model(Ipopt.Optimizer)
@variable(model, x >= 0.0001)
@NLobjective(model, 1 / x)
```
