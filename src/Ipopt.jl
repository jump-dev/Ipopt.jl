# Copyright (c) 2013: Iain Dunning, Miles Lubin, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module Ipopt

import Ipopt_jll
import LinearAlgebra
import MathOptInterface as MOI
import OpenBLAS32_jll

function __init__()
    if VERSION >= v"1.8"
        config = LinearAlgebra.BLAS.lbt_get_config()
        if !any(lib -> lib.interface == :lp64, config.loaded_libs)
            LinearAlgebra.BLAS.lbt_forward(
                OpenBLAS32_jll.libopenblas_path;
                verbose = false,
                clear = false,
            )
        end
    end
    global libipopt = Ipopt_jll.libipopt
    return
end

include("C_wrapper.jl")
include("MOI_wrapper.jl")

export IpoptProblem,
    CreateIpoptProblem,
    FreeIpoptProblem,
    AddIpoptStrOption,
    AddIpoptNumOption,
    AddIpoptIntOption,
    OpenIpoptOutputFile,
    SetIpoptProblemScaling,
    SetIntermediateCallback,
    IpoptSolve

import PrecompileTools

PrecompileTools.@setup_workload begin
    __init__()
    PrecompileTools.@compile_workload begin
        let
            model = MOI.Utilities.CachingOptimizer(
                MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
                MOI.instantiate(Ipopt.Optimizer; with_bridge_type = Float64),
            )
            # We don't want to advertise this option, but it's required so that
            # we don't print the bannner during precompilation.
            MOI.set(model, MOI.RawOptimizerAttribute("sb"), "yes")
            MOI.set(model, MOI.Silent(), true)
            x = MOI.add_variables(model, 3)
            MOI.supports(model, MOI.VariableName(), typeof(x[1]))
            MOI.set(model, MOI.VariableName(), x[1], "x1")
            MOI.set(model, MOI.VariablePrimalStart(), x[1], 0.0)
            for F in (MOI.VariableIndex, MOI.ScalarAffineFunction{Float64})
                MOI.supports_constraint(model, F, MOI.GreaterThan{Float64})
                MOI.supports_constraint(model, F, MOI.LessThan{Float64})
                MOI.supports_constraint(model, F, MOI.EqualTo{Float64})
                # These return false, but it doesn't matter
                MOI.supports_constraint(model, F, MOI.ZeroOne)
                MOI.supports_constraint(model, F, MOI.Integer)
            end
            MOI.add_constraint(model, x[1], MOI.GreaterThan(0.0))
            MOI.add_constraint(model, x[2], MOI.LessThan(0.0))
            MOI.add_constraint(model, x[3], MOI.EqualTo(0.0))
            f = 1.0 * x[1] + x[2] + x[3]
            c1 = MOI.add_constraint(model, f, MOI.GreaterThan(0.0))
            MOI.set(model, MOI.ConstraintName(), c1, "c1")
            MOI.supports(model, MOI.ConstraintName(), typeof(c1))
            MOI.add_constraint(model, f, MOI.LessThan(0.0))
            MOI.add_constraint(model, f, MOI.EqualTo(0.0))
            y, _ = MOI.add_constrained_variables(model, MOI.Nonnegatives(2))
            MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
            MOI.supports(model, MOI.ObjectiveFunction{typeof(f)}())
            MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
            MOI.set(
                model,
                MOI.NLPBlock(),
                MOI.NLPBlockData([], _EmptyNLPEvaluator(), false),
            )
            MOI.optimize!(model)
            MOI.get(model, MOI.TerminationStatus())
            MOI.get(model, MOI.PrimalStatus())
            MOI.get(model, MOI.DualStatus())
            MOI.get(model, MOI.VariablePrimal(), x)
            # We put these after `optimize!` so that the error is thrown on add,
            # not on optimize!
            try
                MOI.add_constraint(model, x[1], MOI.ZeroOne())
            catch
            end
            try
                MOI.add_constraint(model, x[1], MOI.Integer())
            catch
            end
        end
    end
end

end
