module Ipopt

using Libdl
using LinearAlgebra

if VERSION < v"1.3" || (
    haskey(ENV, "JULIA_IPOPT_LIBRARY_PATH") &&
    haskey(ENV, "JULIA_IPOPT_EXECUTABLE_PATH")
)
    const _DEPS_FILE = joinpath(dirname(@__DIR__), "deps", "deps.jl")
    if !isfile(_DEPS_FILE)
        error(
            "Ipopt not properly installed. Please run import Pkg; Pkg.build(\"Ipopt\")",
        )
    end
    include(_DEPS_FILE)
else
    import Ipopt_jll: libipopt
end

function amplexefun(::String)
    return error(
        """
  amplexefun has been removed because it caused problems with other packages.

   - On Julia 1.3 or later, use `Ipopt_jll.amplexe` instead.
   - On Julia 1.0, install Ipopt.jl version 0.6.x via `] add Ipopt@0.6`. Make
     sure to restart Julia for the changes to take effect.
  """,
    )
end

export createProblem,
    addOption,
    openOutputFile,
    setProblemScaling,
    setIntermediateCallback,
    solveProblem,
    IpoptProblem

mutable struct IpoptProblem
    ref::Ptr{Cvoid}             # Reference to the internal data structure
    n::Int                      # Num vars
    m::Int                      # Num cons
    x::Vector{Float64}          # Starting and final solution
    g::Vector{Float64}          # Final constraint values
    mult_g::Vector{Float64}     # lagrange multipliers on constraints
    mult_x_L::Vector{Float64}   # lagrange multipliers on lower bounds
    mult_x_U::Vector{Float64}   # lagrange multipliers on upper bounds
    obj_val::Float64            # Final objective
    status::Int                 # Final status

    # Callbacks
    eval_f::Function
    eval_g::Function
    eval_grad_f::Function
    eval_jac_g::Function
    eval_h::Union{Function,Nothing}
    intermediate::Union{Function,Nothing}

    sense::Symbol               # For MathProgBase

    function IpoptProblem(
        ref::Ptr{Cvoid},
        n,
        m,
        eval_f,
        eval_g,
        eval_grad_f,
        eval_jac_g,
        eval_h,
    )
        prob = new(
            ref,
            n,
            m,
            zeros(Float64, n),
            zeros(Float64, m),
            zeros(Float64, m),
            zeros(Float64, n),
            zeros(Float64, n),
            0.0,
            0,
            eval_f,
            eval_g,
            eval_grad_f,
            eval_jac_g,
            eval_h,
            nothing,
            :Min,
        )
        finalizer(freeProblem, prob)
        return prob
    end
end

# From Ipopt/src/Interfaces/IpReturnCodes_inc.h
const ApplicationReturnStatus = Dict(
    0 => :Solve_Succeeded,
    1 => :Solved_To_Acceptable_Level,
    2 => :Infeasible_Problem_Detected,
    3 => :Search_Direction_Becomes_Too_Small,
    4 => :Diverging_Iterates,
    5 => :User_Requested_Stop,
    6 => :Feasible_Point_Found,
    -1 => :Maximum_Iterations_Exceeded,
    -2 => :Restoration_Failed,
    -3 => :Error_In_Step_Computation,
    -4 => :Maximum_CpuTime_Exceeded,
    -10 => :Not_Enough_Degrees_Of_Freedom,
    -11 => :Invalid_Problem_Definition,
    -12 => :Invalid_Option,
    -13 => :Invalid_Number_Detected,
    -100 => :Unrecoverable_Exception,
    -101 => :NonIpopt_Exception_Thrown,
    -102 => :Insufficient_Memory,
    -199 => :Internal_Error,
)

###########################################################################
# Callback wrappers
###########################################################################

function eval_f_wrapper(
    n::Cint,
    x_ptr::Ptr{Float64},
    new_x::Cint,
    obj_ptr::Ptr{Float64},
    user_data::Ptr{Cvoid},
)
    prob = unsafe_pointer_to_objref(user_data)::IpoptProblem
    x = unsafe_wrap(Array, x_ptr, Int(n))
    new_obj = convert(Float64, prob.eval_f(x))::Float64
    unsafe_store!(obj_ptr, new_obj)
    return Cint(1)  # Return TRUE for success.
end

function eval_g_wrapper(
    n::Cint,
    x_ptr::Ptr{Float64},
    new_x::Cint,
    m::Cint,
    g_ptr::Ptr{Float64},
    user_data::Ptr{Cvoid},
)
    prob = unsafe_pointer_to_objref(user_data)::IpoptProblem
    new_g = unsafe_wrap(Array, g_ptr, Int(m))
    x = unsafe_wrap(Array, x_ptr, Int(n))
    prob.eval_g(x, new_g)
    return Cint(1)  # Return TRUE for success.
end

function eval_grad_f_wrapper(
    n::Cint,
    x_ptr::Ptr{Float64},
    new_x::Cint,
    grad_f_ptr::Ptr{Float64},
    user_data::Ptr{Cvoid},
)
    prob = unsafe_pointer_to_objref(user_data)::IpoptProblem
    new_grad_f = unsafe_wrap(Array, grad_f_ptr, Int(n))
    x = unsafe_wrap(Array, x_ptr, Int(n))
    prob.eval_grad_f(x, new_grad_f)
    return Cint(1)  # Return TRUE for success.
end

function eval_jac_g_wrapper(
    n::Cint,
    x_ptr::Ptr{Float64},
    new_x::Cint,
    m::Cint,
    nele_jac::Cint,
    iRow::Ptr{Cint},
    jCol::Ptr{Cint},
    values_ptr::Ptr{Float64},
    user_data::Ptr{Cvoid},
)
    prob = unsafe_pointer_to_objref(user_data)::IpoptProblem
    # Determine mode
    mode = (values_ptr == C_NULL) ? (:Structure) : (:Values)
    x = unsafe_wrap(Array, x_ptr, Int(n))
    rows = unsafe_wrap(Array, iRow, Int(nele_jac))
    cols = unsafe_wrap(Array, jCol, Int(nele_jac))
    values = unsafe_wrap(Array, values_ptr, Int(nele_jac))
    prob.eval_jac_g(x, mode, rows, cols, values)
    return Cint(1)  # Return TRUE for success.
end

function eval_h_wrapper(
    n::Cint,
    x_ptr::Ptr{Float64},
    new_x::Cint,
    obj_factor::Float64,
    m::Cint,
    lambda_ptr::Ptr{Float64},
    new_lambda::Cint,
    nele_hess::Cint,
    iRow::Ptr{Cint},
    jCol::Ptr{Cint},
    values_ptr::Ptr{Float64},
    user_data::Ptr{Cvoid},
)
    prob = unsafe_pointer_to_objref(user_data)::IpoptProblem
    if prob.eval_h === nothing  # Did the user specify a Hessian?
        return Cint(0)  # Return FALSE for failure.
    end
    # Determine mode
    mode = (values_ptr == C_NULL) ? (:Structure) : (:Values)
    x = unsafe_wrap(Array, x_ptr, Int(n))
    lambda = unsafe_wrap(Array, lambda_ptr, Int(m))
    rows = unsafe_wrap(Array, iRow, Int(nele_hess))
    cols = unsafe_wrap(Array, jCol, Int(nele_hess))
    values = unsafe_wrap(Array, values_ptr, Int(nele_hess))
    prob.eval_h(x, mode, rows, cols, obj_factor, lambda, values)
    return Cint(1)  # Return TRUE for success.
end

function intermediate_wrapper(
    alg_mod::Cint,
    iter_count::Cint,
    obj_value::Float64,
    inf_pr::Float64,
    inf_du::Float64,
    mu::Float64,
    d_norm::Float64,
    regularization_size::Float64,
    alpha_du::Float64,
    alpha_pr::Float64,
    ls_trials::Cint,
    user_data::Ptr{Cvoid},
)
    prob = unsafe_pointer_to_objref(user_data)::IpoptProblem
    ret = prob.intermediate(
        Int(alg_mod),
        Int(iter_count),
        obj_value,
        inf_pr,
        inf_du,
        mu,
        d_norm,
        regularization_size,
        alpha_du,
        alpha_pr,
        Int(ls_trials),
    )
    # Return TRUE if the optimization should continue.
    return ret ? Cint(1) : Cint(0)
end

###########################################################################
# C function wrappers
###########################################################################

function createProblem(
    n::Int,
    x_L::Vector{Float64},
    x_U::Vector{Float64},
    m::Int,
    g_L::Vector{Float64},
    g_U::Vector{Float64},
    nele_jac::Int,
    nele_hess::Int,
    eval_f,
    eval_g,
    eval_grad_f,
    eval_jac_g,
    eval_h = nothing,
)
    @assert n == length(x_L) == length(x_U)
    @assert m == length(g_L) == length(g_U)
    # Wrap callbacks
    eval_f_cb = @cfunction(
        eval_f_wrapper,
        Cint,
        (Cint, Ptr{Float64}, Cint, Ptr{Float64}, Ptr{Cvoid}),
    )
    eval_g_cb = @cfunction(
        eval_g_wrapper,
        Cint,
        (Cint, Ptr{Float64}, Cint, Cint, Ptr{Float64}, Ptr{Cvoid}),
    )
    eval_grad_f_cb = @cfunction(
        eval_grad_f_wrapper,
        Cint,
        (Cint, Ptr{Float64}, Cint, Ptr{Float64}, Ptr{Cvoid}),
    )
    eval_jac_g_cb = @cfunction(
        eval_jac_g_wrapper,
        Cint,
        (
            Cint,
            Ptr{Float64},
            Cint,
            Cint,
            Cint,
            Ptr{Cint},
            Ptr{Cint},
            Ptr{Float64},
            Ptr{Cvoid},
        ),
    )
    eval_h_cb = @cfunction(
        eval_h_wrapper,
        Cint,
        (
            Cint,
            Ptr{Float64},
            Cint,
            Float64,
            Cint,
            Ptr{Float64},
            Cint,
            Cint,
            Ptr{Cint},
            Ptr{Cint},
            Ptr{Float64},
            Ptr{Cvoid},
        ),
    )
    problem_p = ccall(
        (:CreateIpoptProblem, libipopt),
        Ptr{Cvoid},
        (
            Cint,
            Ptr{Float64},
            Ptr{Float64},    # Num vars, var lower and upper bounds
            Cint,
            Ptr{Float64},
            Ptr{Float64},    # Num constraints, con lower and upper bounds
            Cint,
            Cint,                          # Num nnz in constraint Jacobian and in Hessian
            Cint,                                # 0 for C, 1 for Fortran
            Ptr{Cvoid},
            Ptr{Cvoid},              # Callbacks for eval_f, eval_g
            Ptr{Cvoid},
            Ptr{Cvoid},
            Ptr{Cvoid},  # Callbacks for eval_grad_f, eval_jac_g, eval_h
        ),
        n,
        x_L,
        x_U,
        m,
        g_L,
        g_U,
        nele_jac,
        nele_hess,
        1,
        eval_f_cb,
        eval_g_cb,
        eval_grad_f_cb,
        eval_jac_g_cb,
        eval_h_cb,
    )

    if problem_p == C_NULL
        if n == 0
            error(
                "IPOPT: Failed to construct problem because there are 0 " *
                "variables. If you intended to construct an empty problem, " *
                "one work-around is to add a variable fixed to 0.",
            )
        else
            error("IPOPT: Failed to construct problem for some unknown reason.")
        end
    end
    return IpoptProblem(
        problem_p,
        n,
        m,
        eval_f,
        eval_g,
        eval_grad_f,
        eval_jac_g,
        eval_h,
    )
end

# Note: you should have no need to call this directly. `IpoptProblem` has a
# finalizer.
function freeProblem(prob::IpoptProblem)
    if prob.ref == C_NULL
        return
    end
    ccall((:FreeIpoptProblem, libipopt), Cvoid, (Ptr{Cvoid},), prob.ref)
    prob.ref = C_NULL
    return
end

function addOption(prob::IpoptProblem, keyword::String, value::String)
    #/** Function for adding a string option.  Returns FALSE the option
    # *  could not be set (e.g., if keyword is unknown) */
    if !(isascii(keyword) && isascii(value))
        error("IPOPT: Non ASCII parameters not supported")
    end
    ret = ccall(
        (:AddIpoptStrOption, libipopt),
        Cint,
        (Ptr{Cvoid}, Ptr{UInt8}, Ptr{UInt8}),
        prob.ref,
        keyword,
        value,
    )
    if ret == 0
        error("IPOPT: Couldn't set option '$keyword' to value '$value'.")
    end
    return
end

function addOption(prob::IpoptProblem, keyword::String, value::Float64)
    #/** Function for adding a Number option.  Returns FALSE the option
    # *  could not be set (e.g., if keyword is unknown) */
    if !isascii(keyword)
        error("IPOPT: Non ASCII parameters not supported")
    end
    ret = ccall(
        (:AddIpoptNumOption, libipopt),
        Cint,
        (Ptr{Cvoid}, Ptr{UInt8}, Float64),
        prob.ref,
        keyword,
        value,
    )
    if ret == 0
        error("IPOPT: Couldn't set option '$keyword' to value '$value'.")
    end
    return
end

function addOption(prob::IpoptProblem, keyword::String, value::Integer)
    #/** Function for adding an Int option.  Returns FALSE the option
    # *  could not be set (e.g., if keyword is unknown) */
    if !isascii(keyword)
        error("IPOPT: Non ASCII parameters not supported")
    end
    ret = ccall(
        (:AddIpoptIntOption, libipopt),
        Cint,
        (Ptr{Cvoid}, Ptr{UInt8}, Cint),
        prob.ref,
        keyword,
        value,
    )
    if ret == 0
        error("IPOPT: Couldn't set option '$keyword' to value '$value'.")
    end
    return
end

function openOutputFile(prob::IpoptProblem, file_name::String, print_level::Int)
    #/** Function for opening an output file for a given name with given
    # *  printlevel.  Returns false, if there was a problem opening the
    # *  file. */
    if !isascii(file_name)
        error("IPOPT: Non ASCII parameters not supported")
    end
    ret = ccall(
        (:OpenIpoptOutputFile, libipopt),
        Cint,
        (Ptr{Cvoid}, Ptr{UInt8}, Cint),
        prob.ref,
        file_name,
        print_level,
    )
    if ret == 0
        error("IPOPT: Couldn't open output file.")
    end
    return
end

#/** Optional function for setting scaling parameter for the NLP.
# *  This corresponds to the get_scaling_parameters method in TNLP.
# *  If the pointers x_scaling or g_scaling are NULL, then no scaling
# *  for x resp. g is done. */
function setProblemScaling(
    prob::IpoptProblem,
    obj_scaling::Float64,
    x_scaling::Union{Nothing,Vector{Float64}} = nothing,
    g_scaling::Union{Nothing,Vector{Float64}} = nothing,
)
    x_scale_arg = (x_scaling === nothing) ? C_NULL : x_scaling
    g_scale_arg = (g_scaling === nothing) ? C_NULL : g_scaling
    ret = ccall(
        (:SetIpoptProblemScaling, libipopt),
        Cint,
        (Ptr{Cvoid}, Float64, Ptr{Float64}, Ptr{Float64}),
        prob.ref,
        obj_scaling,
        x_scale_arg,
        g_scale_arg,
    )
    if ret == 0
        error("IPOPT: Error setting problem scaling.")
    end
    return
end

function setIntermediateCallback(prob::IpoptProblem, intermediate::Function)
    intermediate_cb = @cfunction(
        intermediate_wrapper,
        Cint,
        (
            Cint,
            Cint,
            Float64,
            Float64,
            Float64,
            Float64,
            Float64,
            Float64,
            Float64,
            Float64,
            Cint,
            Ptr{Cvoid},
        ),
    )
    ret = ccall(
        (:SetIntermediateCallback, libipopt),
        Cint,
        (Ptr{Cvoid}, Ptr{Cvoid}),
        prob.ref,
        intermediate_cb,
    )
    prob.intermediate = intermediate
    if ret == 0
        error("IPOPT: Something went wrong setting the intermediate callback.")
    end
    return
end

function solveProblem(prob::IpoptProblem)
    final_objval = Ref{Cdouble}(0.0)
    ret = ccall(
        (:IpoptSolve, libipopt),
        Cint,
        (
            Ptr{Cvoid},
            Ptr{Float64},
            Ptr{Float64},
            Ptr{Float64},
            Ptr{Float64},
            Ptr{Float64},
            Ptr{Float64},
            Any,
        ),
        prob.ref,
        prob.x,
        prob.g,
        final_objval,
        prob.mult_g,
        prob.mult_x_L,
        prob.mult_x_U,
        prob,
    )
    prob.obj_val = final_objval[]
    prob.status = Int(ret)
    return prob.status
end

include("MOI_wrapper.jl")

end
