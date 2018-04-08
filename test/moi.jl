using MathOptInterface
const MOI = MathOptInterface
const MOIT = MOI.Test
const MOIU = MOI.Utilities
using MathOptInterfaceBridges
const MOIB = MathOptInterfaceBridges

MOIU.@model(IpoptLinearModelData, (), (EqualTo, GreaterThan, LessThan),
           (Zeros, Nonnegatives, Nonpositives, PositiveSemidefiniteConeTriangle),
           (), (SingleVariable,), (ScalarAffineFunction,), (VectorOfVariables,),
           (VectorAffineFunction,))

MOIB.@bridge SplitInterval MOIB.SplitIntervalBridge () (Interval,) () () () (ScalarAffineFunction,) () ()

const optimizer = IpoptOptimizer(print_level=0)
const config = MOIT.TestConfig(atol=1e-4, rtol=1e-4)

@testset "Linear tests" begin
    exclude = ["linear8a", # Behavior in infeasible case doesn't match test.
               "linear12", # Same as above.
               "linear8b", # Behavior in unbounded case doesn't match test.
               "linear8c", # Same as above.
               "linear13", # FeasibilitySense not supported yet.
               "linear7", # VectorAffineFunction not supported.
               "linear1", # SingleVariable-in-EqualTo not supported.
               ]
    linear_optimizer = SplitInterval{Float64}(MOIU.CachingOptimizer(IpoptLinearModelData{Float64}(), optimizer))
    MOIT.contlineartest(linear_optimizer, config, exclude)
end

@testset "MOI NLP tests" begin
    MOIT.nlptest(optimizer, config)
end