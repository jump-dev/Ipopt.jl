using Ipopt

type IpoptProblem
  ref::Ptr{Void}
end

# hs071
# min x1 * x4 * (x1 + x2 + x3) + x3
# st  x1 * x2 * x3 * x4 >= 25
#     x1^2 + x2^2 + x3^2 + x4^2 = 40
#     1 <= x1, x2, x3, x4 <= 5
# Start at (1,5,5,1)
# End at (1.000..., 4.743..., 3.821..., 1.379...)

n = 4
x_L = [1.0, 1.0, 1.0, 1.0]
x_U = [5.0, 5.0, 5.0, 5.0]

m = 2
g_L = [25.0, 40.0]
g_U = [2.0e19, 40.0]

function eval_f(n::Cint, x_ptr::Ptr{Float64}, new_x::Cint, obj_ptr::Ptr{Float64}, user_data::Ptr{Void})
  # Return the value of the objective function at the point $ x$ .
  # n: (in), the number of variables in the problem (dimension of $ x$ ).
  # x: (in), the values for the primal variables, $ x$ , at which  $ f(x)$ is to be evaluated.
  # new_x: (in), false if any evaluation method was previously called with the same values in x, true otherwise.
  # obj_value: (out) the value of the objective function ($ f(x)$ ).
  
  x = pointer_to_array(x_ptr, n)

  new_obj = x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]

  unsafe_store!(obj_ptr, new_obj)

  return 1
end
eval_f_cb = cfunction(eval_f, Cint, (Cint, Ptr{Float64}, Cint, Ptr{Float64}, Ptr{Void}))


function eval_g()

end

function eval_grad_f()

end

function eval_jac_g()

end

function eval_h()

end

ret = ccall((:CreateIpoptProblem, libipopt), Ptr{Void},
        (Cint, Ptr{Float64}, Ptr{Float64},  # Num vars, var lower and upper bounds
         Cint, Ptr{Float64}, Ptr{Float64},  # Num constraints, con lower and upper bounds
         Cint, Cint, # Num nnz in constraint Jacobian and in "Hessian of Lagrangian"
         Cint, # 0 for C, 1 for Fortran
         Any, Any, Any, Any, Any), # Callbacks for eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h
         n, x_L, x_U, m, g_L, g_U, 8, 10, 1,
         eval_f_cb, eval_g_cb, eval_grad_f_cb, eval_jac_g_cb, eval_h_cb)
println(ret)
prob = IpoptProblem(ret)
