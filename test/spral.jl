using JuMP
using Ipopt
using Test

model = Model(Ipopt.Optimizer)
@variable(model, x)
@objective(model, Min, (x - 2)^2)

# Use the linear solver SPRAL
set_attribute(model, "linear_solver", "spral")
optimize!(model)
@test value(x) == 2.0
