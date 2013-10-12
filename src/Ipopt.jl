module Ipopt

  using BinDeps
  @BinDeps.load_dependencies
  
  export libipopt
  export CreateProblem, AddOption, FreeProblem, SolveProblem

  type IpoptProblem
    ref::Ptr{Void}
    n::Int
    x::Vector{Float64}
    obj_val::Float64
    
    function IpoptProblem(ref::Ptr{Void}, n)
      prob = new(ref, n, zeros(Float64, n), 0.0)
      finalizer(prob, FreeProblem)
      prob
    end
  end


  function CreateProblem(n::Int, x_L::Vector{Float64}, x_U::Vector{Float64},
                              m::Int, g_L::Vector{Float64}, g_U::Vector{Float64},
                              nele_jac::Int, nele_hess::Int, 
                              eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h)
    # Wrap callbacks
    eval_f_cb = cfunction(eval_f, Cint, (Cint, Ptr{Float64}, Cint, Ptr{Float64}, Ptr{Void}))
    eval_g_cb = cfunction(eval_g, Cint, (Cint, Ptr{Float64}, Cint, Cint, Ptr{Float64}, Ptr{Void}))
    eval_grad_f_cb = cfunction(eval_grad_f, Cint, (Cint, Ptr{Float64}, Cint, Ptr{Float64}, Ptr{Void}))
    eval_jac_g_cb = cfunction(eval_jac_g, Cint, (Cint, Ptr{Float64}, Cint, Cint, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Float64}, Ptr{Void}))
    eval_h_cb = cfunction(eval_h, Cint, (Cint, Ptr{Float64}, Cint, Float64, Cint, Ptr{Float64}, Cint, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Float64}, Ptr{Void}))

    ret = ccall((:CreateIpoptProblem, libipopt), Ptr{Void},
        (Cint, Ptr{Float64}, Ptr{Float64},  # Num vars, var lower and upper bounds
         Cint, Ptr{Float64}, Ptr{Float64},  # Num constraints, con lower and upper bounds
         Cint, Cint, # Num nnz in constraint Jacobian and in "Hessian of Lagrangian"
         Cint, # 0 for C, 1 for Fortran
         Ptr{Void}, Ptr{Void}, # Callbacks for eval_f, eval_g
         Ptr{Void}, Ptr{Void}, Ptr{Void}), # Callbacks for eval_grad_f, eval_jac_g, eval_h
         n, x_L, x_U, m, g_L, g_U, nele_jac, nele_hess, 0,
         eval_f_cb, eval_g_cb, eval_grad_f_cb, eval_jac_g_cb, eval_h_cb)

    if ret == C_NULL
      error("IPOPT: Failed to construct problem.")
    else
      return(IpoptProblem(ret, n))
    end
  end

  function FreeProblem(prob::IpoptProblem)
    ccall((:FreeIpoptProblem, libipopt), Void, (Ptr{Void},), prob.ref)
  end

  function AddOption(prob::IpoptProblem, keyword::ASCIIString, value::ASCIIString)
    ret = ccall((:AddIpoptStrOption, libipopt), 
                Cint, (Ptr{Void}, Ptr{Uint8}, Ptr{Uint8}),
                prob.ref, keyword, value)
    if ret == 0
      error("IPOPT: Couldn't set option '$keyword' to value '$value'.")
    end
  end

  function AddOption(prob::IpoptProblem, keyword::ASCIIString, value::Float64)
    ret = ccall((:AddIpoptIntOption, libipopt),
                Cint, (Ptr{Void}, Ptr{Uint8}, Float64),
                prob.ref, keyword, value)
    if ret == 0
      error("IPOPT: Couldn't set option '$keyword' to value '$value'.")
    end
  end

  function AddOption(prob::IpoptProblem, keyword::ASCIIString, value::Integer)
    ret = ccall((:AddIpoptIntOption, libipopt),
                Cint, (Ptr{Void}, Ptr{Uint8}, Cint),
                prob.ref, keyword, value)
    if ret == 0
      error("IPOPT: Couldn't set option '$keyword' to value '$value'.")
    end
  end

  # TODO: Expose full functionality
  function SolveProblem(prob::IpoptProblem)
    final_objval = [0.0]
    ret = ccall((:IpoptSolve, libipopt),
                Ptr{Void}, (Ptr{Void}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Void}),
                prob.ref, prob.x, C_NULL, final_objval, C_NULL, C_NULL, C_NULL, C_NULL)

    prob.obj_val = final_objval[1]
  end


end # module
