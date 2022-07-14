# Copyright (c) 2013: Iain Dunning, Miles Lubin, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module TestMOIWrapper

using Ipopt
using Test

const MOI = Ipopt.MOI

function runtests()
    for name in names(@__MODULE__; all = true)
        if startswith("$(name)", "test_")
            @testset "$(name)" begin
                getfield(@__MODULE__, name)()
            end
        end
    end
    return
end

function test_MOI_Test()
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        MOI.Bridges.full_bridge_optimizer(Ipopt.Optimizer(), Float64),
    )
    MOI.set(model, MOI.Silent(), true)
    # Without fixed_variable_treatment set, duals are not computed for variables
    # that have lower_bound == upper_bound.
    MOI.set(
        model,
        MOI.RawOptimizerAttribute("fixed_variable_treatment"),
        "make_constraint",
    )
    MOI.Test.runtests(
        model,
        MOI.Test.Config(
            atol = 1e-4,
            rtol = 1e-4,
            optimal_status = MOI.LOCALLY_SOLVED,
            exclude = Any[
                MOI.ConstraintBasisStatus,
                MOI.DualObjectiveValue,
                MOI.ObjectiveBound,
            ],
        );
        exclude = String[
            # Tests purposefully excluded:
            #  - Upstream: ZeroBridge does not support ConstraintDual
            "test_conic_linear_VectorOfVariables_2",
            #  - Excluded because this test is optional
            "test_model_ScalarFunctionConstantNotZero",
            #  - Excluded because Ipopt returns NORM_LIMIT instead of
            #    DUAL_INFEASIBLE
            "test_solve_TerminationStatus_DUAL_INFEASIBLE",
            #  - Excluded because Ipopt returns INVALID_MODEL instead of
            #    LOCALLY_SOLVED
            "test_linear_VectorAffineFunction_empty_row",
            #  - Excluded because Ipopt returns LOCALLY_INFEASIBLE instead of
            #    INFEASIBLE
            "INFEASIBLE",
            "test_conic_linear_INFEASIBLE",
            "test_conic_linear_INFEASIBLE_2",
            "test_solve_DualStatus_INFEASIBILITY_CERTIFICATE_",
            #  - Excluded due to upstream issue
            "test_model_LowerBoundAlreadySet",
            "test_model_UpperBoundAlreadySet",
            #  - CachingOptimizer does not throw if optimizer not attached
            "test_model_copy_to_UnsupportedAttribute",
            "test_model_copy_to_UnsupportedConstraint",
        ],
    )
    return
end

function test_Name()
    model = Ipopt.Optimizer()
    @test MOI.supports(model, MOI.Name())
    @test MOI.get(model, MOI.Name()) == ""
    MOI.set(model, MOI.Name(), "Model")
    @test MOI.get(model, MOI.Name()) == "Model"
    return
end

function test_ConstraintDualStart()
    model = Ipopt.Optimizer()
    x = MOI.add_variables(model, 2)
    l = MOI.add_constraint(model, x[1], MOI.GreaterThan(1.0))
    u = MOI.add_constraint(model, x[1], MOI.LessThan(1.0))
    e = MOI.add_constraint(model, x[2], MOI.EqualTo(1.0))
    c = MOI.add_constraint(
        model,
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(1.0, x), 0.0),
        MOI.LessThan(1.5),
    )
    @test MOI.get(model, MOI.ConstraintDualStart(), l) === nothing
    @test MOI.get(model, MOI.ConstraintDualStart(), u) === nothing
    @test MOI.get(model, MOI.ConstraintDualStart(), e) === nothing
    @test MOI.get(model, MOI.ConstraintDualStart(), c) === nothing
    @test MOI.get(model, MOI.NLPBlockDualStart()) === nothing
    MOI.set(model, MOI.ConstraintDualStart(), l, 1.0)
    MOI.set(model, MOI.ConstraintDualStart(), u, -1.0)
    MOI.set(model, MOI.ConstraintDualStart(), e, -1.5)
    MOI.set(model, MOI.ConstraintDualStart(), c, 2.0)
    MOI.set(model, MOI.NLPBlockDualStart(), [1.0, 2.0])
    @test MOI.get(model, MOI.ConstraintDualStart(), l) == 1.0
    @test MOI.get(model, MOI.ConstraintDualStart(), u) == -1.0
    @test MOI.get(model, MOI.ConstraintDualStart(), e) == -1.5
    @test MOI.get(model, MOI.ConstraintDualStart(), c) == 2.0
    @test MOI.get(model, MOI.NLPBlockDualStart()) == [1.0, 2.0]
    MOI.set(model, MOI.ConstraintDualStart(), l, nothing)
    MOI.set(model, MOI.ConstraintDualStart(), u, nothing)
    MOI.set(model, MOI.ConstraintDualStart(), e, nothing)
    MOI.set(model, MOI.ConstraintDualStart(), c, nothing)
    MOI.set(model, MOI.NLPBlockDualStart(), nothing)
    @test MOI.get(model, MOI.ConstraintDualStart(), l) === nothing
    @test MOI.get(model, MOI.ConstraintDualStart(), u) === nothing
    @test MOI.get(model, MOI.ConstraintDualStart(), e) === nothing
    @test MOI.get(model, MOI.ConstraintDualStart(), c) === nothing
    @test MOI.get(model, MOI.NLPBlockDualStart()) === nothing
    return
end

function test_solve_time()
    model = Ipopt.Optimizer()
    MOI.add_variable(model)
    @test isnan(MOI.get(model, MOI.SolveTimeSec()))
    MOI.optimize!(model)
    @test MOI.get(model, MOI.SolveTimeSec()) > 0.0
    return
end

# Model structure for test_check_derivatives_for_naninf()
struct Issue136 <: MOI.AbstractNLPEvaluator end
MOI.initialize(::Issue136, ::Vector{Symbol}) = nothing
MOI.features_available(::Issue136) = [:Grad, :Jac]
MOI.eval_objective(::Issue136, x) = x[1]
MOI.eval_constraint(::Issue136, g, x) = (g[1] = x[1]^(1 / 3))
MOI.eval_objective_gradient(::Issue136, grad_f, x) = (grad_f[1] = 1.0)
MOI.jacobian_structure(::Issue136) = Tuple{Int64,Int64}[(1, 1)]
function MOI.eval_constraint_jacobian(::Issue136, J, x)
    J[1] = (1 / 3) * x[1]^(1 / 3 - 1)
    return
end

function test_check_derivatives_for_naninf()
    model = Ipopt.Optimizer()
    x = MOI.add_variable(model)
    MOI.set(
        model,
        MOI.NLPBlock(),
        MOI.NLPBlockData(MOI.NLPBoundsPair.([-Inf], [0.0]), Issue136(), false),
    )
    # Failure to set check_derivatives_for_naninf="yes" may cause Ipopt to
    # segfault or return a NUMERICAL_ERROR status. Check that it is set to "yes"
    # by obtaining an INVALID_MODEL status.
    # MOI.set(model, MOI.RawOptimizerAttribute("check_derivatives_for_naninf"), "no")
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.INVALID_MODEL
    return
end

function test_callback()
    model = Ipopt.Optimizer()
    MOI.set(model, MOI.RawOptimizerAttribute("print_level"), 0)
    x = MOI.add_variable(model)
    MOI.add_constraint(model, x, MOI.GreaterThan(1.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    f = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x)], 0.5)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
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
        push!(x_vals, MOI.get(model, MOI.CallbackVariablePrimal(model), x))
        @test isapprox(obj_value, 1.0 * x_vals[end] + 0.5, atol = 1e-1)
        return iter_count < 1
    end
    MOI.set(model, Ipopt.CallbackFunction(), my_callback)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.INTERRUPTED
    @test length(x_vals) == 2
    @test x_vals[1] !== x_vals[2]
    return
end

function test_callback_2()
    model = Ipopt.Optimizer()
    MOI.set(model, MOI.RawOptimizerAttribute("print_level"), 0)
    x = MOI.add_variable(model)
    MOI.add_constraint(model, x, MOI.GreaterThan(1.0))
    f = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x)], 0.5)
    MOI.add_constraint(model, f, MOI.LessThan(2.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
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
        push!(x_vals, MOI.get(model, MOI.CallbackVariablePrimal(model), x))
        @test isapprox(obj_value, 1.0 * x_vals[end] + 0.5, atol = 1e-1)
        return iter_count < 1
    end
    MOI.set(model, Ipopt.CallbackFunction(), my_callback)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.INTERRUPTED
    @test length(x_vals) == 2
    @test x_vals[1] !== x_vals[2]
    return
end

function test_empty_optimize()
    model = Ipopt.Optimizer()
    @test MOI.get(model, MOI.RawStatusString()) == "Optimize not called"
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.INVALID_MODEL
    @test MOI.get(model, MOI.DualStatus()) == MOI.NO_SOLUTION
    @test MOI.get(model, MOI.PrimalStatus()) == MOI.NO_SOLUTION
    @test MOI.get(model, MOI.RawStatusString()) == "The model has no variable"
    return
end

end  # module TestMOIWrapper

TestMOIWrapper.runtests()
