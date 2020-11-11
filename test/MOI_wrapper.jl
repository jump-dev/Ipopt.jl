module TestMOIWrapper

using Ipopt
using MathOptInterface
using Test

const MOI = MathOptInterface

const OPTIMIZER = Ipopt.Optimizer()
MOI.set(OPTIMIZER, MOI.Silent(), true)

# Without fixed_variable_treatment set, duals are not computed for variables
# that have lower_bound == upper_bound.
MOI.set(
    OPTIMIZER, MOI.RawParameter("fixed_variable_treatment"), "make_constraint"
)

# TODO(odow): add features to Ipopt so we can remove some of this caching.
const BRIDGED_OPTIMIZER = MOI.Bridges.full_bridge_optimizer(
    MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
        OPTIMIZER,
    ),
    Float64,
)

const CONFIG = MOI.Test.TestConfig(
    atol = 1e-4,
    rtol = 1e-4,
    optimal_status = MOI.LOCALLY_SOLVED,
    infeas_certificates = false,
)

const CONFIG_NO_DUAL = MOI.Test.TestConfig(
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
    MOI.Test.unittest(
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
        ]
    )
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
        ]
    )
    MOI.Test.linear1test(BRIDGED_OPTIMIZER, CONFIG_NO_DUAL)
    MOI.Test.linear2test(BRIDGED_OPTIMIZER, CONFIG_NO_DUAL)
    MOI.Test.linear10test(BRIDGED_OPTIMIZER, CONFIG_NO_DUAL)
    MOI.Test.linear14test(BRIDGED_OPTIMIZER, CONFIG_NO_DUAL)
end

function test_qp()
    MOI.Test.qptest(BRIDGED_OPTIMIZER, CONFIG)
end

function test_qcp()
    MOI.empty!(BRIDGED_OPTIMIZER)
    MOI.Test.qcptest(BRIDGED_OPTIMIZER, CONFIG_NO_DUAL)
end

function test_nlptest()
    MOI.Test.nlptest(OPTIMIZER, CONFIG)
end

function test_getters()
    MOI.Test.copytest(
        MOI.instantiate(Ipopt.Optimizer, with_bridge_type=Float64),
        MOI.Utilities.Model{Float64}()
    )
end

function test_boundsettwice()
    MOI.Test.set_lower_bound_twice(OPTIMIZER, Float64)
    MOI.Test.set_upper_bound_twice(OPTIMIZER, Float64)
end

function test_nametest()
    MOI.Test.nametest(BRIDGED_OPTIMIZER)
end

function test_validtest()
    MOI.Test.validtest(BRIDGED_OPTIMIZER)
end

function test_emptytest()
    MOI.Test.emptytest(BRIDGED_OPTIMIZER)
end

function test_solve_time()
    model = Ipopt.Optimizer()
    x = MOI.add_variable(model)
    @test MOI.get(model, MOI.SolveTime()) == NaN
    MOI.optimize!(model)
    @test MOI.get(model, MOI.SolveTime()) > 0.0
end

end  # module TestMOIWrapper

runtests(TestMOIWrapper)
