# Ipopt.jl

![](https://www.coin-or.org/wordpress/wp-content/uploads/2014/08/COINOR.png)

**Ipopt.jl** is a [Julia](http://julialang.org/) interface to the [COIN-OR](www.coin-or.org)
nonlinear solver [Ipopt](http://www.coin-or.org/Ipopt/documentation/documentation.html).

*Note: This wrapper is maintained by the JuMP community and is not a COIN-OR
project.*
[![Build Status](https://github.com/jump-dev/Ipopt.jl/workflows/CI/badge.svg?branch=master)](https://github.com/jump-dev/Ipopt.jl/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/jump-dev/Ipopt.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jump-dev/Ipopt.jl)

## Installation

Install `Ipopt.jl` using the Julia package manager:
```julia
import Pkg; Pkg.add("Ipopt")
```

In addition to installing the `Ipopt.jl` package, this will also download and
install the Ipopt binaries. You do _not_ need to install Ipopt separately.

If you require a custom build of Ipopt, see the instructions below.

For details on using a different linear solver, see the `Linear Solvers` section
below.

## JuMP and MathOptInterface

You can use Ipopt with JuMP as follows:
```julia
using JuMP, Ipopt
model = Model(Ipopt.Optimizer)
set_optimizer_attribute(model, "max_cpu_time", 60.0)
set_optimizer_attribute(model, "print_level", 0)
```

Supported options are listed in the [Ipopt documentation](https://coin-or.github.io/Ipopt/OPTIONS.html#OPTIONS_REF).

## C Interface Wrapper

Full documentation for the Ipopt C wrapper is available [here](http://ipoptjl.readthedocs.org/en/latest/ipopt.html);
however, we strongly recommend you use Ipopt via JuMP.

## `INVALID_MODEL` error

If you get a termination status `MOI.INVALID_MODEL`, it is probably because you
have some undefined value in your model, e.g., a division by zero. Fix this by
removing the division, or by imposing variable bounds so that you cut off the
undefined region.

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

## Custom Installation

**Note: it is not necessary to compile a custom version of Ipopt to use a
different linear solver. See the Linear Solvers section below.**

To install custom built Ipopt binaries set the environmental variables
`JULIA_IPOPT_LIBRARY_PATH` and `JULIA_IPOPT_EXECUTABLE_PATH`, and call
`import Pkg; Pkg.build("Ipopt")`. For instance, if the libraries are installed
in `/opt/lib` and the executable is in `/opt/bin` just call
```julia
ENV["JULIA_IPOPT_LIBRARY_PATH"] = "/opt/lib"
ENV["JULIA_IPOPT_EXECUTABLE_PATH"] = "/opt/bin"
import Pkg; Pkg.build("Ipopt")
```

If you do not want BinaryProvider to download the default binaries on install
set  `JULIA_IPOPT_LIBRARY_PATH` and `JULIA_IPOPT_EXECUTABLE_PATH`  before
calling `import Pkg; Pkg.add("Ipopt")`.

To switch back to the default binaries clear `JULIA_IPOPT_LIBRARY_PATH` and
`JULIA_IPOPT_EXECUTABLE_PATH`, and call `import Pkg; Pkg.build("Ipopt")`.

## Linear Solvers

To improve performance, Ipopt supports a number of linear solvers. Installing
these can be tricky.

### Pardiso Project

#### Linux

Tested on a clean install of Ubuntu 20.04.

1. Install lapack and libomp:
   ```
   sudo apt install liblapack3 libomp-dev
   ```
2. Download Pardiso from [https://www.pardiso-project.org](https://www.pardiso-project.org)
3. Rename the file `libpardiso-XXXXX.so` to `libpardiso.so`, and place it
   somewhere on your load path.
   - Alternatively, if the library is located at `/full/path/libpardiso.dylib`,
     start Julia with `export LD_LIBRARY_PATH=/full/path; julia`
4. Set the option `linear_solver` to `pardiso`:
   ```julia
   using Libdl
   # Note: these filenames may differ. Check `/usr/lib/x86_64-linux-gnu` for the
   # specific extension.
   Libdl.dlopen("/usr/lib/x86_64-linux-gnu/liblapack.so.3", RTLD_GLOBAL)
   Libdl.dlopen("/usr/lib/x86_64-linux-gnu/libomp.so.5", RTLD_GLOBAL)

   using JuMP, Ipopt
   model = Model(Ipopt.Optimizer)
   set_optimizer_attribute(model, "linear_solver", "pardiso")
   ```

#### Mac

1. Download Pardiso from [https://www.pardiso-project.org](https://www.pardiso-project.org)
2. Rename the file `libpardiso-XXXXX.dylib` to `libpardiso.dylib`, and place it
   somewhere on your load path.
   - Alternatively, if the library is located at `/full/path/libpardiso.dylib`,
     start Julia with `export LD_LOAD_PATH=/full/path; julia`
4. Set the option `linear_solver` to `pardiso`:
   ```julia
   using JuMP, Ipopt
   model = Model(Ipopt.Optimizer)
   set_optimizer_attribute(model, "linear_solver", "pardiso")
   ```

#### Windows

Currently untested. If you have instructions that work, please open an issue.

### MA27

#### Linux

Currently untested. If you have instructions that work, please open an issue.

#### Mac

1. Download HSL for IPOPT from http://www.hsl.rl.ac.uk/ipopt/
2. Unzip the download, and run the following:
   ```
   ./configure --prefix=</full/path/somewhere>
   make
   ```
   where `</full/path/somewhere>` is replaced as appropriate.
3. Rename the files `/full/path/somewhere/lib/libcoinhsl.dylib` to
   `/full/path/somewhere/lib/libhsl.dylib`, and place the library somewhere on
   your load path.
   - Alternatively, start Julia with `export LD_LOAD_PATH=/full/path/somewhere/lib; julia`
4. Set the option `linear_solver` to `ma27`:
   ```julia
   using JuMP, Ipopt
   model = Model(Ipopt.Optimizer)
   set_optimizer_attribute(model, "linear_solver", "ma27")
   ```

#### Windows

Currently untested. If you have instructions that work, please open an issue.
