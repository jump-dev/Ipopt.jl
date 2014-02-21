module Ipopt

if isfile(joinpath(Pkg.dir("Ipopt"),"deps","deps.jl"))
    include("../deps/deps.jl")
else
    error("Ipopt not properly installed. Please run Pkg.build(\"Ipopt\")")
end
  
  export createProblem, addOption
  export openOutputFile, setProblemScaling, setIntermediateCallback
  export solveProblem
  export IpoptProblem


  type IpoptProblem
    ref::Ptr{Void}  # Reference to the internal data structure
    n::Int  # Num vars
    m::Int  # Num cons
    x::Vector{Float64}  # Starting and final solution
    g::Vector{Float64}  # Final constraint values
    obj_val::Float64  # Final objective
    status::Int  # Final status
    
    # Callbacks
    eval_f::Function
    eval_g::Function
    eval_grad_f::Function
    eval_jac_g::Function
    eval_h  # Can be nothing
    intermediate  # Can be nothing

    # For MathProgBase
    sense::Symbol

    function IpoptProblem(
      ref::Ptr{Void}, n, m,
      eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h)
      prob = new(ref, n, m, zeros(Float64, n), zeros(Float64, m), 0.0, 0,
                 eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h, nothing,
                 :Min)
      # Free the internal IpoptProblem structure when
      # the Julia IpoptProblem instance goes out of scope
      finalizer(prob, freeProblem)
      # Return the object we just made
      prob
    end
  end


  ApplicationReturnStatus = {
    0=>:Solve_Succeeded,
    1=>:Solved_To_Acceptable_Level,
    2=>:Infeasible_Problem_Detected,
    3=>:Search_Direction_Becomes_Too_Small,
    4=>:Diverging_Iterates,
    5=>:User_Requested_Stop,
    6=>:Feasible_Point_Found,
    -1=>:Maximum_Iterations_Exceeded,
    -2=>:Restoration_Failed,
    -3=>:Error_In_Step_Computation,
    -4=>:Maximum_CpuTime_Exceeded,
    -10=>:Not_Enough_Degrees_Of_Freedom,
    -11=>:Invalid_Problem_Definition,
    -12=>:Invalid_Option,
    -13=>:Invalid_Number_Detected,
    -100=>:Unrecoverable_Exception,
    -101=>:NonIpopt_Exception_Thrown,
    -102=>:Insufficient_Memory,
    -199=>:Internal_Error}


  ###########################################################################
  # Callback wrappers
  ###########################################################################
  # Objective (eval_f)
  function eval_f_wrapper(n::Cint, x_ptr::Ptr{Float64}, new_x::Cint, obj_ptr::Ptr{Float64}, user_data::Ptr{Void})
    # Extract Julia the problem from the pointer
    prob = unsafe_pointer_to_objref(user_data)::IpoptProblem
    # Calculate the new objective
    new_obj = convert(Float64, prob.eval_f(pointer_to_array(x_ptr, int(n))))
    # Fill out the pointer
    unsafe_store!(obj_ptr, new_obj)
    # Done
    return int32(1)
  end

  # Constraints (eval_g)
  function eval_g_wrapper(n::Cint, x_ptr::Ptr{Float64}, new_x::Cint, m::Cint, g_ptr::Ptr{Float64}, user_data::Ptr{Void})
    # Extract Julia the problem from the pointer
    prob = unsafe_pointer_to_objref(user_data)::IpoptProblem
    # Calculate the new constraint values
    new_g = pointer_to_array(g_ptr, int(m))
    prob.eval_g(pointer_to_array(x_ptr, int(n)), new_g)
    # Done
    return int32(1)
  end

  # Objective gradient (eval_grad_f)
  function eval_grad_f_wrapper(n::Cint, x_ptr::Ptr{Float64}, new_x::Cint, grad_f_ptr::Ptr{Float64}, user_data::Ptr{Void})
    # Extract Julia the problem from the pointer
    prob = unsafe_pointer_to_objref(user_data)::IpoptProblem
    # Calculate the gradient
    new_grad_f = pointer_to_array(grad_f_ptr, int(n))
    prob.eval_grad_f(pointer_to_array(x_ptr, int(n)), new_grad_f)
    # Done
    return int32(1)
  end

  # Jacobian (eval_jac_g)
  function eval_jac_g_wrapper(n::Cint, x_ptr::Ptr{Float64}, new_x::Cint, m::Cint, nele_jac::Cint, iRow::Ptr{Cint}, jCol::Ptr{Cint}, values::Ptr{Float64}, user_data::Ptr{Void})
    # Extract Julia the problem from the pointer
    prob = unsafe_pointer_to_objref(user_data)::IpoptProblem
    # Determine mode
    mode = (values == C_NULL) ? (:Structure) : (:Values)
    x = pointer_to_array(x_ptr, int(n))
    rows = pointer_to_array(iRow, int(nele_jac))
    cols = pointer_to_array(jCol, int(nele_jac))
    values = pointer_to_array(values, int(nele_jac))
    prob.eval_jac_g(x, mode, rows, cols, values)
    # Done
    return int32(1)
  end

  # Hessian
  function eval_h_wrapper(n::Cint, x_ptr::Ptr{Float64}, new_x::Cint, obj_factor::Float64, m::Cint, lambda_ptr::Ptr{Float64}, new_lambda::Cint, nele_hess::Cint, iRow::Ptr{Cint}, jCol::Ptr{Cint}, values::Ptr{Float64}, user_data::Ptr{Void})
    # Extract Julia the problem from the pointer
    prob = unsafe_pointer_to_objref(user_data)::IpoptProblem
    # Did the user specify a Hessian
    if prob.eval_h == nothing
      # No Hessian provided
      return int32(0)
    else
      # Determine mode
      mode = (values == C_NULL) ? (:Structure) : (:Values)
      x = pointer_to_array(x_ptr, int(n))
      lambda = pointer_to_array(lambda_ptr, int(m))
      rows = pointer_to_array(iRow, int(nele_hess))
      cols = pointer_to_array(jCol, int(nele_hess))
      values = pointer_to_array(values, int(nele_hess))
      prob.eval_h(x, mode, rows, cols, obj_factor, lambda, values)
      # Done
      return int32(1)
    end
  end

  # Intermediate
  function intermediate_wrapper(alg_mod::Cint, iter_count::Cint, obj_value::Float64, inf_pr::Float64, inf_du::Float64, mu::Float64, d_norm::Float64, regularization_size::Float64, alpha_du::Float64, alpha_pr::Float64, ls_trials::Cint, user_data::Ptr{Void})
    # Extract Julia the problem from the pointer
    prob = unsafe_pointer_to_objref(user_data)::IpoptProblem
    keepgoing = prob.intermediate(int(alg_mod), int(iter_count), obj_value, inf_pr, inf_du, mu, d_norm, regularization_size, alpha_du, alpha_pr, int(ls_trials))
    # Done
    return keepgoing ? int32(1) : int32(0)
  end

  ###########################################################################
  # C function wrappers
  ###########################################################################
  function createProblem(n::Int, x_L::Vector{Float64}, x_U::Vector{Float64},
                         m::Int, g_L::Vector{Float64}, g_U::Vector{Float64},
                         nele_jac::Int, nele_hess::Int, 
                         eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h = nothing)
    # Wrap callbacks
    eval_f_cb = cfunction(eval_f_wrapper, Cint,
                        (Cint, Ptr{Float64}, Cint, Ptr{Float64}, Ptr{Void}))
    eval_g_cb = cfunction(eval_g_wrapper, Cint,
                        (Cint, Ptr{Float64}, Cint, Cint, Ptr{Float64}, Ptr{Void}))
    eval_grad_f_cb = cfunction(eval_grad_f_wrapper, Cint,
                        (Cint, Ptr{Float64}, Cint, Ptr{Float64}, Ptr{Void}))
    eval_jac_g_cb = cfunction(eval_jac_g_wrapper, Cint,
                        (Cint, Ptr{Float64}, Cint, Cint, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Float64}, Ptr{Void}))
    eval_h_cb = cfunction(eval_h_wrapper, Cint, 
                        (Cint, Ptr{Float64}, Cint, Float64, Cint, Ptr{Float64}, Cint, Cint, Ptr{Cint}, Ptr{Cint}, Ptr{Float64}, Ptr{Void}))

    ret = ccall((:CreateIpoptProblem, libipopt), Ptr{Void},
        (Cint, Ptr{Float64}, Ptr{Float64},  # Num vars, var lower and upper bounds
         Cint, Ptr{Float64}, Ptr{Float64},  # Num constraints, con lower and upper bounds
         Cint, Cint,                        # Num nnz in constraint Jacobian and in Hessian
         Cint,                              # 0 for C, 1 for Fortran
         Ptr{Void}, Ptr{Void},              # Callbacks for eval_f, eval_g
         Ptr{Void}, Ptr{Void}, Ptr{Void}),  # Callbacks for eval_grad_f, eval_jac_g, eval_h
         n, x_L, x_U, m, g_L, g_U, nele_jac, nele_hess, 1,
         eval_f_cb, eval_g_cb, eval_grad_f_cb, eval_jac_g_cb, eval_h_cb)

    if ret == C_NULL
      error("IPOPT: Failed to construct problem.")
    else
      return(IpoptProblem(ret, n, m, eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h))
    end
  end

  # TODO: Not even expose this? Seems dangerous, should just destruct
  # the IpoptProblem object via GC
  function freeProblem(prob::IpoptProblem)
    ccall((:FreeIpoptProblem, libipopt), Void, (Ptr{Void},), prob.ref)
  end


  function addOption(prob::IpoptProblem, keyword::ASCIIString, value::ASCIIString)
    #/** Function for adding a string option.  Returns FALSE the option
    # *  could not be set (e.g., if keyword is unknown) */
    ret = ccall((:AddIpoptStrOption, libipopt), 
                Cint, (Ptr{Void}, Ptr{Uint8}, Ptr{Uint8}),
                prob.ref, keyword, value)
    if ret == 0
      error("IPOPT: Couldn't set option '$keyword' to value '$value'.")
    end
  end


  function addOption(prob::IpoptProblem, keyword::ASCIIString, value::Float64)
    #/** Function for adding a Number option.  Returns FALSE the option
    # *  could not be set (e.g., if keyword is unknown) */
    ret = ccall((:AddIpoptNumOption, libipopt),
                Cint, (Ptr{Void}, Ptr{Uint8}, Float64),
                prob.ref, keyword, value)
    if ret == 0
      error("IPOPT: Couldn't set option '$keyword' to value '$value'.")
    end
  end


  function addOption(prob::IpoptProblem, keyword::ASCIIString, value::Integer)
    #/** Function for adding an Int option.  Returns FALSE the option
    # *  could not be set (e.g., if keyword is unknown) */
    ret = ccall((:AddIpoptIntOption, libipopt),
                Cint, (Ptr{Void}, Ptr{Uint8}, Cint),
                prob.ref, keyword, value)
    if ret == 0
      error("IPOPT: Couldn't set option '$keyword' to value '$value'.")
    end
  end


  function openOutputFile(prob::IpoptProblem, file_name::ASCIIString, print_level::Int)
    #/** Function for opening an output file for a given name with given
    # *  printlevel.  Returns false, if there was a problem opening the
    # *  file. */
    ret = ccall((:OpenIpoptOutputFile, libipopt),
                Cint, (Ptr{Void}, Ptr{Uint8}, Cint),
                prob.ref, file_name, print_level)
    if ret == 0
      error("IPOPT: Couldn't open output file.")
    end
  end

  # TODO: Verify this function even works! Trying it with 0.5 on HS071
  # seems to change nothing.
  function setProblemScaling(prob::IpoptProblem, obj_scaling::Float64,
                             x_scaling = nothing,
                             g_scaling = nothing)
    #/** Optional function for setting scaling parameter for the NLP.
    # *  This corresponds to the get_scaling_parameters method in TNLP.
    # *  If the pointers x_scaling or g_scaling are NULL, then no scaling
    # *  for x resp. g is done. */
    x_scale_arg = (x_scaling == nothing) ? C_NULL : x_scaling
    g_scale_arg = (g_scaling == nothing) ? C_NULL : g_scaling
    ret = ccall((:SetIpoptProblemScaling, libipopt),
                Cint, (Ptr{Void}, Float64, Ptr{Float64}, Ptr{Float64}),
                prob.ref, obj_scaling, x_scale_arg, g_scale_arg)
    if ret == 0
      error("IPOPT: Error setting problem scaling.")
    end
  end

  
  function setIntermediateCallback(prob::IpoptProblem, intermediate::Function)
    intermediate_cb = cfunction(intermediate_wrapper, Cint,
                        (Cint, Cint, Float64, Float64, Float64, Float64,
                        Float64, Float64, Float64, Float64, Cint, Ptr{Void}))
    ret = ccall((:SetIntermediateCallback, libipopt), Cint,
                (Ptr{Void}, Ptr{Void}), prob.ref, intermediate_cb)
    prob.intermediate = intermediate
    if ret == 0
      error("IPOPT: Something went wrong setting the intermediate callback.")
    end
  end


  function solveProblem(prob::IpoptProblem)
    final_objval = [0.0]
    ret = ccall((:IpoptSolve, libipopt),
                Cint, (Ptr{Void}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Any),
                prob.ref, prob.x, prob.g, final_objval, C_NULL, C_NULL, C_NULL, prob)
    prob.obj_val = final_objval[1]
    
    return int(ret)
  end

  function solveProblem(prob::IpoptProblem, mult_g::Vector{Float64}, mult_x_L::Vector{Float64}, mult_x_U::Vector{Float64})
    final_objval = [0.0]
    ret = ccall((:IpoptSolve, libipopt),
                Cint, (Ptr{Void}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Ptr{Float64}, Any),
                prob.ref, prob.x, prob.g, final_objval, mult_g, mult_x_L, mult_x_U, prob)
    prob.obj_val = final_objval[1]
    prob.status = int(ret)

    return int(ret)
  end


  include("IpoptSolverInterface.jl")

end # module
