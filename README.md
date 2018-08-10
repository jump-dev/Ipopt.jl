Ipopt.jl
========

[![Build Status](https://travis-ci.org/JuliaOpt/Ipopt.jl.svg?branch=master)](https://travis-ci.org/JuliaOpt/Ipopt.jl)
[![Coverage Status](https://img.shields.io/coveralls/JuliaOpt/Ipopt.jl.svg)](https://coveralls.io/r/JuliaOpt/Ipopt.jl)

[![Ipopt](http://pkg.julialang.org/badges/Ipopt_0.7.svg)](http://pkg.julialang.org/?pkg=Ipopt&ver=0.7)
[![Ipopt](http://pkg.julialang.org/badges/Ipopt_0.7.svg)](http://pkg.julialang.org/?pkg=Ipopt&ver=1.0)

**Ipopt.jl** is a [Julia](http://julialang.org/) interface to the [Ipopt](http://www.coin-or.org/Ipopt/documentation/documentation.html) nonlinear solver.

**Default Installation**: `julia> Pkg.add("Ipopt")`

This will install Ipopt.jl, as well as Ipopt itself. A binary will be downloaded
by default on macOS or Windows, and Ipopt will be automatically built from source
on Linux unless a pre-existing version is found on the `LD_LIBRARY_PATH`.
If your platform is not supported, or if you prefer to compile your own version
of Ipopt in order to use commercial sparse linear algebra libraries, use
the instructions below.

**Custom Installation**:

Make sure you have the required packages before installing, e.g.,

```bash
sudo apt-get install build-essential gfortran pkg-config liblapack-dev libblas-dev
```

The script below was tested successfully for installing Ipopt. You may modify
the configuration options, but be sure to install Ipopt into the correct
prefix location.

```bash
wget https://www.coin-or.org/download/source/Ipopt/Ipopt-3.12.8.tgz
tar xvzf Ipopt-3.12.8.tgz
cd Ipopt-3.12.8/
# Blas and Lapack must be installed already. If not, run
# ThirdParty/Blas/get.Blas and ThirdParty/Lapack/get.Lapack.
# ASL is required even if you do not plan to use it.
cd ThirdParty/ASL/
./get.ASL
cd ..
cd Mumps
# Compiling Mumps requires gfortran.
./get.Mumps
cd ../..
# Update the prefix location! The following is correct only for Julia 0.6.
./configure --prefix=$HOME/.julia/v0.6/Ipopt/deps/usr
make
make test
make install
```

Now in Julia:

```julia
julia> Pkg.build("Ipopt")
INFO: Building Ipopt

julia> Pkg.test("Ipopt")
...
INFO: Ipopt tests passed
```

MathProgBase Interface
----------------------

Ipopt implements the solver-independent [MathProgBase](https://github.com/JuliaOpt/MathProgBase.jl) interface,
and so can be used within modeling software like [JuMP](https://github.com/JuliaOpt/JuMP.jl).
The solver object is called ``IpoptSolver``. All options listed in the [Ipopt documentation](http://www.coin-or.org/Ipopt/documentation/node40.html) may be passed directly. For example, you can suppress output by saying ``IpoptSolver(print_level=0)``. If you wish to pass an option specifically for the restoration phase, instead of using the prefix ``resto.``, use the prefix ``resto_``. For example ``IpoptSolver(resto_max_iter=0)``.

C Interface Wrapper
-------------------

Full documentation for the Ipopt C wrapper is available [here](http://ipoptjl.readthedocs.org/en/latest/ipopt.html). Use of the [nonlinear MathProgBase interface](http://mathprogbasejl.readthedocs.org/en/latest/nlp.html) is recommended over the low-level C interface because it permits one to easily switch between solvers.
