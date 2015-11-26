using Ipopt
using Base.Test

# First of all, test that hs071 example works
include("hs071_test.jl")

# Test setting some options
# String option
println("\nString option")
addOption(prob, "hessian_approximation", "exact")
@test_throws ErrorException addOption(prob, "hessian_approximation", "badoption")
println("\nInt option")
# Int option
addOption(prob, "file_print_level", 3) == nothing
@test_throws ErrorException addOption(prob, "file_print_level", -1)
# Double option
println("\nFloat option")
addOption(prob, "derivative_test_tol", 0.5)
@test_throws ErrorException addOption(prob, "derivative_test_tol", -1.0)

# Test opening an output file
openOutputFile(prob, "blah.txt", 5)

# Test MathProgBase stuff
#include(joinpath(Pkg.dir("MathProgBase"),"test","linprog.jl"))
#linprogtest(IpoptSolver())
    using MathProgBase
    solver = IpoptSolver()
    sol = linprog([-1,0],[2 1],'<',1.5,solver)
    @test sol.status == :Optimal
    @test abs(sol.objval - -0.75) <= 1e-6
    @test norm(sol.sol - [0.75,0.0]) <= 1e-6

    sol = linprog([-1,0],sparse([2 1]),'<',1.5,solver)
    @test sol.status == :Optimal
    @test abs(sol.objval - -0.75) <= 1e-6
    @test norm(sol.sol - [0.75,0.0]) <= 1e-6

    # test infeasible problem:
    # min x
    # s.t. 2x+y <= -1
    # x,y >= 0
    sol = linprog([1,0],[2 1],'<',-1,solver)
#    @test sol.status == :Infeasible

    # test unbounded problem:
    # min -x-y
    # s.t. -x+2y <= 0
    # x,y >= 0
    sol = linprog([-1,-1],[-1 2],'<',[0],solver)
#    @test sol.status == :Unbounded


include(joinpath(Pkg.dir("MathProgBase"),"test","nlp.jl"))
nlptest(IpoptSolver())
nlptest_nohessian(IpoptSolver())
convexnlptest(IpoptSolver())
rosenbrocktest(IpoptSolver())

include(joinpath(Pkg.dir("MathProgBase"),"test","quadprog.jl"))
quadprogtest(IpoptSolver())
qpdualtest(IpoptSolver())

# Test retoration only options
#
# Warm start with infeasible solution, force restoration on initial iteration. 
# But limit to 0 iterations. Forces :UserLimit exit.
m = MathProgBase.NonlinearModel(IpoptSolver(start_with_resto="yes", resto_max_iter=0))
l = [1,1,1,1]
u = [5,5,5,5]
lb = [25, 40]
ub = [Inf, 40]
MathProgBase.loadproblem!(m, 4, 2, l, u, lb, ub, :Min, HS071())
MathProgBase.setwarmstart!(m,[0,15,15,11])
MathProgBase.optimize!(m)
@test MathProgBase.status(m) == :UserLimit
