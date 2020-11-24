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

### Solver-specific callback

Ipopt provides a callback that can be used to log the status of the optimization
during a solve. It can also be used to terminate the optimization by returning
`false`. Here is an example:

```julia
using JuMP, Ipopt, Test
model = Model(Ipopt.Optimizer)
set_silent(model)
@variable(model, x >= 1)
@objective(model, Min, x + 0.5)
x_vals = Float64[]
function my_callback(
   prob::IpoptProblem,
   alg_mod::Cint,
   iter_count::Cint,
   obj_value::Float64,
   inf_pr::Float64,
   inf_du::Float64,
   mu::Float64,
   d_norm::Float64,
   regularization_size::Float64,
   alpha_du::Float64,
   alpha_pr::Float64,
   ls_trials::Cint,
)
   c = Ipopt.column(index(x))
   push!(x_vals, prob.x[c])
   @test isapprox(obj_value, 1.0 * x_vals[end] + 0.5, atol = 1e-1)
   # return `true` to keep going, or `false` to terminate the optimization.
   return iter_count < 1
end
MOI.set(model, Ipopt.CallbackFunction(), my_callback)
optimize!(model)
@test MOI.get(model, MOI.TerminationStatus()) == MOI.INTERRUPTED
@test length(x_vals) == 2
```
See the [Ipopt documentation](https://coin-or.github.io/Ipopt/OUTPUT.html) for
an explanation of the arguments to the callback. They are identical to the
output contained in the logging table printed to the screen.

## C Interface Wrapper

Full documentation for the Ipopt C wrapper [is available](https://github.com/jump-dev/Ipopt.jl/blob/master/doc/C_API.md).
However, we strongly encourage you to use Ipopt with JuMP instead.

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

To install custom built Ipopt binaries, you must compile the shared library (
e.g., `libipopt.dylib`, `libipopt.so`, or `libipopt.dll`) _and_ the AMPL
executable (e.g., `ipopt` or `ipopt.exe`).

If you cannot compile the AMPL executable, you can [download an appropriate
version from AMPL](https://ampl.com/products/solvers/open-source/#ipopt).

Next, set the environmental variables `JULIA_IPOPT_LIBRARY_PATH` and
`JULIA_IPOPT_EXECUTABLE_PATH` to point the the shared library and AMPL
executable repspectively. Then call `import Pkg; Pkg.build("Ipopt")`.

For instance, given `/Users/oscar/lib/libipopt.dylib` and
`/Users/oscar/bin/ipopt`, run:
```julia
ENV["JULIA_IPOPT_LIBRARY_PATH"] = "/Users/oscar/lib"
ENV["JULIA_IPOPT_EXECUTABLE_PATH"] = "/Users/oscar/bin"
import Pkg
Pkg.build("Ipopt")
```

**Very important note: you must set these environment variables before
calling `using Ipopt` in every Julia session.**

For example:
```julia
ENV["JULIA_IPOPT_LIBRARY_PATH"] = "/Users/oscar/lib"
ENV["JULIA_IPOPT_EXECUTABLE_PATH"] = "/Users/oscar/bin"
using Ipopt
```
Alternatively, you can set these permanently through your operating system.

To switch back to the default binaries, run
```julia
delete!(ENV, "JULIA_IPOPT_LIBRARY_PATH")
delete!(ENV, "JULIA_IPOPT_EXECUTABLE_PATH")
import Pkg
Pkg.build("Ipopt")
```

## Linear Solvers

To improve performance, Ipopt supports a number of linear solvers. Installing
these can be tricky, however, the following instructions should work. If they
don't, or are not explicit enough, please open an issue.

### Pardiso (Pardiso Project)

#### Linux

_Tested on a clean install of Ubuntu 20.04._

1. Install lapack and libomp:
   ```
   sudo apt install liblapack3 libomp-dev
   ```
2. Download Pardiso from [https://www.pardiso-project.org](https://www.pardiso-project.org)
3. Rename the file `libpardiso-XXXXX.so` to `libpardiso.so`
4. Place the `libpardiso.so` library somewhere on your load path.
   - Alternatively, if the library is located at `/full/path/libpardiso.dylib`,
     start Julia with `export LD_LIBRARY_PATH=/full/path; julia`
5. Set the option `linear_solver` to `pardiso`:
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

_Tested on a MacBook Pro, 10.15.7._

1. Download Pardiso from [https://www.pardiso-project.org](https://www.pardiso-project.org)
2. Rename the file `libpardiso-XXXXX.dylib` to `libpardiso.dylib`.
3. Place the `libpardiso.dylib` library somewhere on your load path.
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

### Pardiso (MKL)

#### Linux

Currently untested. If you have instructions that work, please open an issue.

#### Mac

Currently untested. If you have instructions that work, please open an issue.

#### Windows

Currently untested. If you have instructions that work, please open an issue.

### HSL (MA27)

#### Linux

_Tested on a clean install of Ubuntu 20.04._

1. Install Fortran compiler if necessary
   ```
   sudo apt install gfortran
   ```
2. Download HSL for IPOPT from http://www.hsl.rl.ac.uk/ipopt/
3. Unzip the download, and run the following:
   ```
   ./configure --prefix=</full/path/somewhere>
   make
   make install
   ```
   where `</full/path/somewhere>` is replaced as appropriate.
4. Rename the file `/full/path/somewhere/lib/libcoinhsl.so` to
   `/full/path/somewhere/lib/libhsl.so`.
5. Place the `libhsl.so` library somewhere on your load path.
   - Alternatively, start Julia with `export LD_LIBRARY_PATH=/full/path/somewhere/lib; julia`
6. Set the option `linear_solver` to `ma27`:
   ```julia
   using JuMP, Ipopt
   model = Model(Ipopt.Optimizer)
   set_optimizer_attribute(model, "linear_solver", "ma27")
   ```

#### Mac

_Tested on a MacBook Pro, 10.15.7._

1. Download HSL for IPOPT from http://www.hsl.rl.ac.uk/ipopt/
2. Unzip the download, and run the following:
   ```
   ./configure --prefix=</full/path/somewhere>
   make
   make install
   ```
   where `</full/path/somewhere>` is replaced as appropriate.
3. Rename the file `/full/path/somewhere/lib/libcoinhsl.dylib` to
   `/full/path/somewhere/lib/libhsl.dylib`
4. Place the `libhsl.dylib` library somewhere on your load path.
   - Alternatively, start Julia with `export LD_LOAD_PATH=/full/path/somewhere/lib; julia`
5. Set the option `linear_solver` to `ma27`:
   ```julia
   using JuMP, Ipopt
   model = Model(Ipopt.Optimizer)
   set_optimizer_attribute(model, "linear_solver", "ma27")
   ```

#### Windows

Currently untested. If you have instructions that work, please open an issue.

### HSL (MA86, MA97)

#### Linux

Currently untested. If you have instructions that work, please open an issue.

#### Mac

Currently untested. If you have instructions that work, please open an issue.

#### Windows

Currently untested. If you have instructions that work, please open an issue.
