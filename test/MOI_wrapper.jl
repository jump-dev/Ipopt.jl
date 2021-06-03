module TestMOIWrapper

using Ipopt
using MathOptInterface
using Test

const MOI = MathOptInterface

function _optimizer()
    model = Ipopt.Optimizer()
    MOI.set(model, MOI.Silent(), true)
    # Without fixed_variable_treatment set, duals are not computed for variables
    # that have lower_bound == upper_bound.
    MOI.set(
        model,
        MOI.RawOptimizerAttribute("fixed_variable_treatment"),
        "make_constraint",
    )
    return model
end

const OPTIMIZER = _optimizer()

# TODO(odow): add features to Ipopt so we can remove some of this caching.
const BRIDGED_OPTIMIZER = MOI.Bridges.full_bridge_optimizer(
    MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        _optimizer(),
    ),
    Float64,
)

const CONFIG = MOI.Test.Config(
    atol = 1e-4,
    rtol = 1e-4,
    optimal_status = MOI.LOCALLY_SOLVED,
    infeas_certificates = false,
)

const CONFIG_NO_DUAL = MOI.Test.Config(
    atol = 1e-4,
    rtol = 1e-4,
    optimal_status = MOI.LOCALLY_SOLVED,
    infeas_certificates = false,
    duals = false,  # Don't check dual result!
)

function test_solvername()
    @test MOI.get(OPTIMIZER, MOI.SolverName()) == "Ipopt"
end

function test_supports_default_copy_to()
    @test MOI.Utilities.supports_default_copy_to(OPTIMIZER, false)
    @test !MOI.Utilities.supports_default_copy_to(OPTIMIZER, true)
end

function test_basicconstraint()
    # TODO(odow): these basic constraint tests are broken.
    # MOI.Test.basic_constraint_tests(OPTIMIZER, CONFIG)
end

function test_unittest()
    return MOI.Test.unittest(
        BRIDGED_OPTIMIZER,
        CONFIG,
        String[
            # VectorOfVariables-in-SecondOrderCone not supported
            "delete_soc_variables",
            # NumberOfThreads not supported
            "number_threads",
            # MOI.Integer not supported.
            "solve_integer_edge_cases",
            # ObjectiveBound not supported.
            "solve_objbound_edge_cases",
            # DualObjectiveValue not supported.
            "solve_result_index",
            # Returns NORM_LIMIT instead of DUAL_INFEASIBLE
            "solve_unbounded_model",
            # MOI.ZeroOne not supported.
            "solve_zero_one_with_bounds_1",
            "solve_zero_one_with_bounds_2",
            "solve_zero_one_with_bounds_3",
        ],
    )
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
end

function test_contlinear()
    MOI.Test.contlineartest(
        BRIDGED_OPTIMIZER,
        CONFIG,
        String[
            # Tests requiring DualObjectiveValue. Tested below.
            "linear1",
            "linear2",
            "linear10",
            "linear14",
            # Tests requiring infeasibility certificates
            "linear8a",
            "linear8b",
            "linear8c",
            "linear12",
            # An INVALID_MODEL because it contains an empty 0 == 0 row.
            "linear15",
        ],
    )
    MOI.Test.linear1test(BRIDGED_OPTIMIZER, CONFIG_NO_DUAL)
    MOI.Test.linear2test(BRIDGED_OPTIMIZER, CONFIG_NO_DUAL)
    MOI.Test.linear10test(BRIDGED_OPTIMIZER, CONFIG_NO_DUAL)
    return MOI.Test.linear14test(BRIDGED_OPTIMIZER, CONFIG_NO_DUAL)
end

function test_qp()
    return MOI.Test.qptest(BRIDGED_OPTIMIZER, CONFIG)
end

function test_qcp()
    MOI.Test.qcptest(BRIDGED_OPTIMIZER, CONFIG_NO_DUAL)
    return
end

function test_nlptest()
    return MOI.Test.nlptest(OPTIMIZER, CONFIG)
end

function test_getters()
    return MOI.Test.copytest(
        MOI.instantiate(Ipopt.Optimizer, with_bridge_type = Float64),
        MOI.Utilities.Model{Float64}(),
    )
end

function test_boundsettwice()
    MOI.Test.set_lower_bound_twice(OPTIMIZER, Float64)
    return MOI.Test.set_upper_bound_twice(OPTIMIZER, Float64)
end

function test_nametest()
    return MOI.Test.nametest(BRIDGED_OPTIMIZER)
end

function test_validtest()
    return MOI.Test.validtest(BRIDGED_OPTIMIZER)
end

function test_emptytest()
    return MOI.Test.emptytest(BRIDGED_OPTIMIZER)
end

function test_solve_time()
    model = Ipopt.Optimizer()
    x = MOI.add_variable(model)
    @test isnan(MOI.get(model, MOI.SolveTimeSec()))
    MOI.optimize!(model)
    @test MOI.get(model, MOI.SolveTimeSec()) > 0.0
end

# Model structure for test_check_derivatives_for_naninf()
struct Issue136 <: MOI.AbstractNLPEvaluator end
MOI.initialize(::Issue136, ::Vector{Symbol}) = nothing
MOI.features_available(d::Issue136) = [:Grad, :Jac]
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
end

function test_deprecation()
    model = Ipopt.Optimizer(print_level = 0)
    @test MOI.get(model, MOI.RawOptimizerAttribute("print_level")) == 0
end

function test_callback()
    model = Ipopt.Optimizer()
    MOI.set(model, MOI.RawOptimizerAttribute("print_level"), 0)
    x = MOI.add_variable(model)
    MOI.add_constraint(model, MOI.SingleVariable(x), MOI.GreaterThan(1.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    f = MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x)], 0.5)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
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
        c = Ipopt.column(x)
        push!(x_vals, prob.x[c])
        @test isapprox(obj_value, 1.0 * x_vals[end] + 0.5, atol = 1e-1)
        return iter_count < 1
    end
    MOI.set(model, Ipopt.CallbackFunction(), my_callback)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.INTERRUPTED
    @test length(x_vals) == 2
end

function test_empty_optimize()
    model = Ipopt.Optimizer()
    err = ErrorException(
        "IPOPT: Failed to construct problem because there are 0 variables. " *
        "If you intended to construct an empty problem, one work-around is " *
        "to add a variable fixed to 0.",
    )
    @test_throws err MOI.optimize!(model)
end

end  # module TestMOIWrapper

runtests(TestMOIWrapper)
