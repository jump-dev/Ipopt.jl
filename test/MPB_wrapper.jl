# Test MathProgBase stuff

using MathProgBase
using LinearAlgebra
using SparseArrays

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

function mathprogbase_file(file::String)
    return joinpath(dirname(dirname(pathof(MathProgBase))), "test", file)
end


include(mathprogbase_file("nlp.jl"))
nlptest(IpoptSolver())
nlptest_nohessian(IpoptSolver())
convexnlptest(IpoptSolver())
rosenbrocktest(IpoptSolver())

include(mathprogbase_file("quadprog.jl"))
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
