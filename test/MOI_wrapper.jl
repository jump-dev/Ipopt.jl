using MathOptInterface
const MOI = MathOptInterface
const MOIT = MOI.Test
const MOIU = MOI.Utilities
const MOIB = MOI.Bridges

MOIU.@model(IpoptModelData,
            (),
            (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan),
            (MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives),
            (),
            (MOI.SingleVariable,),
            (MOI.ScalarAffineFunction, MOI.ScalarQuadraticFunction),
            (MOI.VectorOfVariables,),
            (MOI.VectorAffineFunction,))

# Without fixed_variable_treatment set, duals are not computed for variables
# that have lower_bound == upper_bound.
const optimizer = Ipopt.Optimizer(print_level=0, fixed_variable_treatment="make_constraint")
const config = MOIT.TestConfig(atol=1e-4, rtol=1e-4,
                               optimal_status=MOI.LOCALLY_SOLVED)

@testset "MOI Linear tests" begin
    exclude = ["linear8a", # Behavior in infeasible case doesn't match test.
               "linear12", # Same as above.
               "linear8b", # Behavior in unbounded case doesn't match test.
               "linear8c", # Same as above.
               "linear7",  # VectorAffineFunction not supported.
               "linear15", # VectorAffineFunction not supported.
               ]
    model_for_ipopt = MOIU.UniversalFallback(IpoptModelData{Float64}())
    linear_optimizer = MOI.Bridges.SplitInterval{Float64}(
                         MOIU.CachingOptimizer(model_for_ipopt, optimizer))
    MOIT.contlineartest(linear_optimizer, config, exclude)
end

MOI.empty!(optimizer)

@testset "MOI QP/QCQP tests" begin
    qp_optimizer = MOIU.CachingOptimizer(IpoptModelData{Float64}(), optimizer)
    MOIT.qptest(qp_optimizer, config)
    exclude = ["qcp1", # VectorAffineFunction not supported.
              ]
    MOIT.qcptest(qp_optimizer, config, exclude)
end

MOI.empty!(optimizer)

@testset "MOI NLP tests" begin
    MOIT.nlptest(optimizer, config)
end
