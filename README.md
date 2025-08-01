![](https://www.coin-or.org/wordpress/wp-content/uploads/2014/08/COINOR.png)

# Ipopt.jl

[![Build Status](https://github.com/jump-dev/Ipopt.jl/actions/workflows/ci.yml/badge.svg?branch=master)](https://github.com/jump-dev/Ipopt.jl/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/jump-dev/Ipopt.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/jump-dev/Ipopt.jl)

[Ipopt.jl](https://github.com/jump-dev/Ipopt.jl) is a wrapper for the
[Ipopt](https://github.com/coin-or/ipopt) solver.

## Affiliation

This wrapper is maintained by the JuMP community and is not a COIN-OR project.

## Getting help

If you need help, please ask a question on the [JuMP community forum](https://jump.dev/forum).

If you have a reproducible example of a bug, please [open a GitHub issue](https://github.com/jump-dev/Ipopt.jl/issues/new).

## License

`Ipopt.jl` is licensed under the [MIT License](https://github.com/jump-dev/Ipopt.jl/blob/master/LICENSE.md).

The underlying solver, [coin-or/Ipopt](https://github.com/coin-or/Ipopt), is
licensed under the [Eclipse public license](https://github.com/coin-or/Ipopt/blob/master/LICENSE).

## Installation

Install `Ipopt.jl` using the Julia package manager:
```julia
import Pkg
Pkg.add("Ipopt")
```

In addition to installing the `Ipopt.jl` package, this will also download and
install the Ipopt binaries. You do not need to install Ipopt separately.

To use a custom binary, read the [Custom solver binaries](https://jump.dev/JuMP.jl/stable/developers/custom_solver_binaries/)
section of the JuMP documentation.

For details on using a different linear solver, see the `Linear Solvers` section
below. You do not need a custom binary to change the linear solver.

## Use with JuMP

You can use Ipopt with JuMP as follows:
```julia
using JuMP, Ipopt
model = Model(Ipopt.Optimizer)
set_attribute(model, "max_cpu_time", 60.0)
set_attribute(model, "print_level", 0)
```

### Type stability

Ipopt.jl v1.10.0 moved the `Ipopt.Optimizer` object to a package extension. As a
consequence, `Ipopt.Optimizer` is now type unstable, and it will be inferred as
`Ipopt.Optimizer()::Any`.

In most cases, this should not impact performance. If it does, there are two
work-arounds.

First, you can use a function barrier:
```julia
using JuMP, Ipopt
function main(optimizer::T) where {T}
   model = Model(optimizer)
   return
end
main(Ipopt.Optimizer)
```
Although the outer `Ipopt.Optimizer` is type unstable, the `optimizer` inside
`main` will be properly inferred.

Second, you may explicitly get and use the extension module:
```julia
using JuMP, Ipopt
const IpoptMathOptInterfaceExt =
   Base.get_extension(Ipopt, :IpoptMathOptInterfaceExt)
model = Model(IpoptMathOptInterfaceExt.Optimizer)
```

## MathOptInterface API

The Ipopt optimizer supports the following constraints and attributes.

List of supported objective functions:

 * [`MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}`](@ref)
 * [`MOI.ObjectiveFunction{MOI.ScalarNonlinearFunction}`](@ref)
 * [`MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}`](@ref)
 * [`MOI.ObjectiveFunction{MOI.VariableIndex}`](@ref)

List of supported variable types:

 * [`MOI.Reals`](@ref)
 * [`MOI.Parameter{Float64}`](@ref)

List of supported constraint types:

 * [`MOI.ScalarAffineFunction{Float64}`](@ref) in [`MOI.EqualTo{Float64}`](@ref)
 * [`MOI.ScalarAffineFunction{Float64}`](@ref) in [`MOI.GreaterThan{Float64}`](@ref)
 * [`MOI.ScalarAffineFunction{Float64}`](@ref) in [`MOI.Interval{Float64}`](@ref)
 * [`MOI.ScalarAffineFunction{Float64}`](@ref) in [`MOI.LessThan{Float64}`](@ref)
 * [`MOI.ScalarNonlinearFunction`](@ref) in [`MOI.EqualTo{Float64}`](@ref)
 * [`MOI.ScalarNonlinearFunction`](@ref) in [`MOI.GreaterThan{Float64}`](@ref)
 * [`MOI.ScalarNonlinearFunction`](@ref) in [`MOI.Interval{Float64}`](@ref)
 * [`MOI.ScalarNonlinearFunction`](@ref) in [`MOI.LessThan{Float64}`](@ref)
 * [`MOI.ScalarQuadraticFunction{Float64}`](@ref) in [`MOI.EqualTo{Float64}`](@ref)
 * [`MOI.ScalarQuadraticFunction{Float64}`](@ref) in [`MOI.GreaterThan{Float64}`](@ref)
 * [`MOI.ScalarQuadraticFunction{Float64}`](@ref) in [`MOI.Interval{Float64}`](@ref)
 * [`MOI.ScalarQuadraticFunction{Float64}`](@ref) in [`MOI.LessThan{Float64}`](@ref)
 * [`MOI.VariableIndex`](@ref) in [`MOI.EqualTo{Float64}`](@ref)
 * [`MOI.VariableIndex`](@ref) in [`MOI.GreaterThan{Float64}`](@ref)
 * [`MOI.VariableIndex`](@ref) in [`MOI.Interval{Float64}`](@ref)
 * [`MOI.VariableIndex`](@ref) in [`MOI.LessThan{Float64}`](@ref)

List of supported model attributes:

 * [`MOI.BarrierIterations`](@ref)
 * [`MOI.NLPBlock`](@ref)
 * [`MOI.NLPBlockDualStart`](@ref)
 * [`MOI.Name`](@ref)
 * [`MOI.ObjectiveSense`](@ref)
 * [`MOI.SolveTimeSec`](@ref)

List of supported optimizer attributes:

 * [`MOI.AutomaticDifferentiationBackend`](@ref)

## Options

Supported options are listed in the [Ipopt documentation](https://coin-or.github.io/Ipopt/OPTIONS.html#OPTIONS_REF).

## Solver-specific callbacks

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
   push!(x_vals, callback_value(model, x))
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

To access the current solution and primal, dual, and complementarity violations
of each iteration, use `Ipopt.GetIpoptCurrentViolations` and
`Ipopt.GetIpoptCurrentIterate`. The two functions are identical to the ones in
the [Ipopt C interface](https://coin-or.github.io/Ipopt/INTERFACES.html).

## C API

Ipopt.jl wraps the [Ipopt C interface](https://coin-or.github.io/Ipopt/INTERFACES.html)
with minimal modifications.

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

!!! warning
    If `values === nothing`, `x` is an undefined object. Accessing any elements
    in it will cause Julia to segfault.
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

!!! warning
    If `values === nothing`, `x` is an undefined object. Accessing any elements
    in it will cause Julia to segfault.
"""
function eval_h end
```

## `INVALID_MODEL` error

If you get a termination status `MOI.INVALID_MODEL`, it is probably because you
have some undefined value in your model, for example, a division by zero. Fix
this by removing the division, or by imposing variable bounds so that you cut
off the undefined region.

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

## Linear Solvers

To improve performance, Ipopt supports a number of linear solvers.

### HSL

Obtain a license and download `HSL_jll.jl` from [
https://licences.stfc.ac.uk/products/Software/HSL/LibHSL](https://licences.stfc.ac.uk/products/Software/HSL/LibHSL).

Install this download into your current environment using:
```julia
import Pkg
Pkg.develop(path = "/full/path/to/HSL_jll.jl")
```

Then, use a linear solver in HSL by setting the `hsllib` and `linear_solver`
attributes:
```julia
using JuMP, Ipopt
import HSL_jll
model = Model(Ipopt.Optimizer)
set_attribute(model, "hsllib", HSL_jll.libhsl_path)
set_attribute(model, "linear_solver", "ma86")
```

The available HSL solvers are `"ma27"`, `"ma57"`, `"ma77"`, `"ma86"`, and `"ma97"`.
We recommend using either sequential BLAS and LAPACK backends or a multithreaded version
limited to one thread when employing the linear solvers `"ma86"`, or `"ma97"`.
These solvers already leverage parallelism via OpenMP, and enabling multiple threads in
BLAS and LAPACK may result in thread oversubscription.

#### macOS users

Due to the security policy of macOS, Mac users may need to delete the quarantine
attribute of the ZIP archive before extracting. For example:
```raw
xattr -d com.apple.quarantine HSL_jll.jl-2025.7.21.zip
```

### Pardiso

Download Pardiso from [https://www.pardiso-project.org](https://www.pardiso-project.org).
Save the shared library somewhere, and record the filename.

Then, use Pardiso by setting the `pardisolib` and `linear_solver` attributes:
```julia
using JuMP, Ipopt
model = Model(Ipopt.Optimizer)
set_attribute(model, "pardisolib", "/full/path/to/libpardiso")
set_attribute(model, "linear_solver", "pardiso")
```

### SPRAL

If you use Ipopt.jl with Julia ≥ v1.9, the linear solver [SPRAL](https://github.com/ralna/spral) is available.
You can use it by setting the `linear_solver` attribute:
```julia
using JuMP, Ipopt
model = Model(Ipopt.Optimizer)
set_attribute(model, "linear_solver", "spral")
```
Note that the following environment variables must be set before starting Julia:
```raw
export OMP_CANCELLATION=TRUE
export OMP_PROC_BIND=TRUE
```

## BLAS and LAPACK

With Julia v1.9 or later, Ipopt and the linear solvers [MUMPS](https://mumps-solver.org/index.php)
(default), SPRAL, and HSL are compiled with
[`libblastrampoline`](https://github.com/JuliaLinearAlgebra/libblastrampoline)
(LBT), a library that can change between BLAS and LAPACK backends at runtime.

Note that the BLAS and LAPACK backends loaded at runtime must be compiled with 32-bit integers.
The default BLAS and LAPACK backend is [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS),
and we rely on the Julia artifact `OpenBLAS32_jll.jl` if no backend is loaded before `using Ipopt`.

Using LBT, we can also switch dynamically to other BLAS backends such as Intel
MKL, BLIS, and Apple Accelerate. Because Ipopt and the linear solvers heavily
rely on BLAS and LAPACK routines, using an optimized backend for a particular
platform can improve the performance.

### Sequential BLAS and LAPACK

If you have `ReferenceBLAS32_jll.jl` and `LAPACK32_jll.jl` installed,
switch to sequential and [reference version of BLAS and LAPACK](https://github.com/Reference-LAPACK/lapack) with:
```julia
using ReferenceBLAS32_jll, LAPACK32_jll
LinearAlgebra.BLAS.lbt_forward(libblas32)
LinearAlgebra.BLAS.lbt_forward(liblapack32)
using Ipopt
```

### MKL

If you have [MKL.jl](https://github.com/JuliaLinearAlgebra/MKL.jl) installed,
switch to MKL by adding `using MKL` to your code:

```julia
using MKL
using Ipopt
```

### BLIS

If you have `BLIS32_jll.jl` and `LAPACK32_jll.jl` installed,
switch to [BLIS](https://github.com/flame/blis) with:

```julia
using blis32_jll, LAPACK32_jll
LinearAlgebra.BLAS.lbt_forward(blis32)
LinearAlgebra.BLAS.lbt_forward(liblapack32)
using Ipopt
```

### AppleAccelerate

If you are using macOS ≥ v13.4 and you have [AppleAccelerate.jl](https://github.com/JuliaLinearAlgebra/AppleAccelerate.jl) installed, add `using AppleAccelerate` to your code:

```julia
using AppleAccelerate
using Ipopt
```

### Display backends

Check what backends are loaded using:
```julia
import LinearAlgebra
LinearAlgebra.BLAS.lbt_get_config()
```
