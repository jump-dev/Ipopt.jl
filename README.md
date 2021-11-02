**Ipopt.jl 0.9 contains breaking changes to the C API.**

This release of Ipopt.jl contains a number of breaking changes, however, we
anticipate that this will be the last breaking change before Ipopt v1.0.

 * The MathProgBase wrapper has been removed
 * The C API has been refactored in a breaking way:
   * All functions are now named the same as their C counterparts
   * `addOption` has been removed in favor of explicit calls to
      `AddIpoptStrOption`, `AddIpoptIntOption`, or `AddIpoptNumOption`
   * The jacobian and hessian callbacks no longer take a `mode::Symbol`
     argument. Instead, the `values` is `nothing` if the structure is requested.

# Ipopt.jl

![](https://www.coin-or.org/wordpress/wp-content/uploads/2014/08/COINOR.png)

**Ipopt.jl** is a [Julia](http://julialang.org/) interface to the [COIN-OR](https://www.coin-or.org)
nonlinear solver [Ipopt](https://coin-or.github.io/Ipopt/).

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

Ipopt.jl wraps the [Ipopt C interface](https://coin-or.github.io/Ipopt/INTERFACES.html) with minimal modifications.

A complete example is available in the `test/C_wrapper.jl` file.

For simplicity, the five callbacks required by Ipopt are slightly different to
the C interface. They are as follows:
```julia
"""
   eval_f(x::Vector{Float64})::Float64

Returns the objective value `f(x)`.
"""
function eval_f end

"""
   eval_grad_f(x::Vector{Float64}, grad_f::Vector{Float64})::Nothing

Fills `grad_f` in-place with the gradient of the objective function evaluated at
`x`.
"""
function eval_grad_f end

"""
   eval_g(x::Vector{Float64}, g::Vector{Float64})::Nothing

Fills `g` in-place with the value of the constraints evaluated at `x`.
"""
function eval_g end

"""
   eval_jac_g(
      x::Vector{Float64},
      rows::Vector{Cint},
      cols::Vector{Cint},
      values::Union{Nothing,Vector{Float64}},
   )::Nothing

Compute the Jacobian matrix.

* If `values === nothing`
   - Fill `rows` and `cols` with the 1-indexed sparsity structure
* Otherwise:
   - Fill `values` with the elements of the Jacobian matrix according to the
     sparsity structure.
"""
function eval_jac_g end

"""
   eval_h(
      x::Vector{Float64},
      rows::Vector{Cint},
      cols::Vector{Cint},
      obj_factor::Float64,
      lambda::Float64,
      values::Union{Nothing,Vector{Float64}},
   )::Nothing

Compute the Hessian-of-the-Lagrangian matrix.

* If `values === nothing`
   - Fill `rows` and `cols` with the 1-indexed sparsity structure
* Otherwise:
   - Fill `values` with the Hessian matrix according to the sparsity structure.
"""
function eval_h end
```

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
4. Place the `libpardiso.so` library somewhere on your load path
   - Alternatively, if the library is located at `/full/path/libpardiso.so`,
     start Julia with `export LD_LIBRARY_PATH=/full/path; julia`

     To make this permanent, modify your `.bashrc` to include:
     ```
     export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/full/path/"
     ```
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
     start Julia with `export DL_LOAD_PATH=/full/path; julia`
4. Set the option `linear_solver` to `pardiso`:
   ```julia
   using JuMP, Ipopt
   model = Model(Ipopt.Optimizer)
   set_optimizer_attribute(model, "linear_solver", "pardiso")
   ```

#### Windows

Currently untested. If you have instructions that work, please open an issue.

### HSL (MA27, MA86, MA97)

#### Linux

_Tested on a clean install of Ubuntu 20.04._

1. Install Fortran compiler if necessary
   ```
   sudo apt install gfortran
   ```
2. Download the appropriate version of HSL.
   - MA27: [HSL for IPOPT from HSL](http://www.hsl.rl.ac.uk/ipopt/)
   - MA86: [HSL_MA86 from HSL](http://www.hsl.rl.ac.uk/download/HSL_MA86/1.6.0/)
   - Other: http://www.hsl.rl.ac.uk/catalogue/
3. Unzip the download, `cd` to the directory, and run the following:
   ```
   ./configure --prefix=</full/path/somewhere>
   make
   make install
   ```
   where `</full/path/somewhere>` is replaced as appropriate.
4. Rename the resutling HSL library to `/full/path/somewhere/lib/libhsl.so`.
   - For `ma27`, the file is `/full/path/somewhere/lib/libcoinhsl.so`
   - For `ma86`, the file is `/full/path/somewhere/lib/libhsl_ma86.so`
5. Place the `libhsl.so` library somewhere on your load path.
   - Alternatively, start Julia with `export LD_LIBRARY_PATH=/full/path/somewhere/lib; julia`
6. Set the option `linear_solver` to `ma27` or `ma86` as appropriate:
   ```julia
   using JuMP, Ipopt
   model = Model(Ipopt.Optimizer)
   set_optimizer_attribute(model, "linear_solver", "ma27")
   # or
   set_optimizer_attribute(model, "linear_solver", "ma86")
   ```

#### Mac

_Tested on a MacBook Pro, 10.15.7._

1. Download the appropriate version of HSL.
   - MA27: [HSL for IPOPT from HSL](http://www.hsl.rl.ac.uk/ipopt/)
   - MA86: [HSL_MA86 from HSL](http://www.hsl.rl.ac.uk/download/HSL_MA86/1.6.0/)
   - Other: http://www.hsl.rl.ac.uk/catalogue/
2. Unzip the download, `cd` to the directory, and run the following:
   ```
   ./configure --prefix=</full/path/somewhere>
   make
   make install
   ```
   where `</full/path/somewhere>` is replaced as appropriate.
3. Rename the resutling HSL library to `/full/path/somewhere/lib/libhsl.dylib`.
   - For `ma27`, the file is `/full/path/somewhere/lib/libcoinhsl.dylib`
   - For `ma86`, the file is `/full/path/somewhere/lib/libhsl_ma86.dylib`
4. Place the `libhsl.dylib` library somewhere on your load path.
   - Alternatively, start Julia with `export DL_LOAD_PATH=/full/path/somewhere/lib; julia`
5. Set the option `linear_solver` to `ma27` or `ma86` as appropriate:
   ```julia
   using JuMP, Ipopt
   model = Model(Ipopt.Optimizer)
   set_optimizer_attribute(model, "linear_solver", "ma27")
   # or
   set_optimizer_attribute(model, "linear_solver", "ma86")
   ```

#### Windows

Currently untested. If you have instructions that work, please open an issue.

### Pardiso (MKL)

Currently untested on all platforms. If you have instructions that work, please
open an issue.
