using Ipopt

# hs071
# min x1 * x4 * (x1 + x2 + x3) + x3
# st  x1 * x2 * x3 * x4 >= 25
#     x1^2 + x2^2 + x3^2 + x4^2 = 40
#     1 <= x1, x2, x3, x4 <= 5
# Start at (1,5,5,1)
# End at (1.000..., 4.743..., 3.821..., 1.379...)

function eval_f(prob::IpoptProblem, x::Vector{Float64}) 
  return x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
end

function eval_g(prob::IpoptProblem, x::Vector{Float64}, g::Vector{Float64})
  # Bad: g    = zeros(2)  # Allocates new array
  # OK:  g[:] = zeros(2)  # Modifies 'in place'
  g[1] = x[1]   * x[2]   * x[3]   * x[4]
  g[2] = x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2
end

function eval_grad_f(prob::IpoptProblem, x::Vector{Float64}, grad_f::Vector{Float64})
  # Bad: grad_f    = zeros(4)  # Allocates new array
  # OK:  grad_f[:] = zeros(4)  # Modifies 'in place'
  grad_f[1] = x[1] * x[4] + x[4] * (x[1] + x[2] + x[3])
  grad_f[2] = x[1] * x[4]
  grad_f[3] = x[1] * x[4] + 1
  grad_f[4] = x[1] * (x[1] + x[2] + x[3])
end

function eval_jac_g(prob::IpoptProblem, x::Vector{Float64}, mode, rows::Vector{Int32}, cols::Vector{Int32}, values::Vector{Float64})
  if mode == :Structure
    # Constraint (row) 1
    rows[1] = 1; cols[1] = 1
    rows[2] = 1; cols[2] = 2
    rows[3] = 1; cols[3] = 3
    rows[4] = 1; cols[4] = 4
    # Constraint (row) 2
    rows[5] = 2; cols[5] = 1
    rows[6] = 2; cols[6] = 2
    rows[7] = 2; cols[7] = 3
    rows[8] = 2; cols[8] = 4
  else
    # Constraint (row) 1
    values[1] = x[2]*x[3]*x[4]  # 1,1
    values[2] = x[1]*x[3]*x[4]  # 1,2
    values[3] = x[1]*x[2]*x[4]  # 1,3
    values[4] = x[1]*x[2]*x[3]  # 1,4
    # Constraint (row) 2
    values[5] = 2*x[1]  # 2,1
    values[6] = 2*x[2]  # 2,2
    values[7] = 2*x[3]  # 2,3
    values[8] = 2*x[4]  # 2,4
  end
end


function eval_h(n::Cint, x_ptr::Ptr{Float64}, new_x::Cint, obj_factor::Float64, m::Cint, lamda::Ptr{Float64}, new_lambda::Cint, nele_hess::Cint, iRow::Ptr{Cint}, jCol::Ptr{Cint}, values::Ptr{Float64}, user_data::Ptr{Void})
  println("eval h")
  # Do it later
  return int32(0)
end


function intermediate(alg_mod::Cint, iter_count::Cint, 
  obj_value::Float64, inf_pr::Float64, inf_du::Float64, mu::Float64,
  d_norm::Float64, regularization_size::Float64, alpha_du::Float64, alpha_pr::Float64, 
  ls_trials::Cint, user_data::Ptr{Void})
  println("Intermediate")
  println(alg_mod)
  println(obj_value)
  return int32(1)
end

n = 4
x_L = [1.0, 1.0, 1.0, 1.0]
x_U = [5.0, 5.0, 5.0, 5.0]

m = 2
g_L = [25.0, 40.0]
g_U = [2.0e19, 40.0]

prob = CreateProblem(n, x_L, x_U, m, g_L, g_U, 8, 10,
                     eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h)
AddOption(prob, "hessian_approximation", "limited-memory")
OpenOutputFile(prob, "test.txt", 2)
SetProblemScaling(prob, 0.5)
SetIntermediateCallback(prob, intermediate)
prob.x = [1.0, 5.0, 5.0, 1.0]
status = SolveProblem(prob)
println(Ipopt.ApplicationReturnStatus[status])

println(prob.x)
println(prob.obj_val)
