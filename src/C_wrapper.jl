# Copyright (c) 2013: Iain Dunning, Miles Lubin, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

mutable struct IpoptProblem{M}
    ipopt_problem::Ptr{Cvoid}   # Reference to the internal data structure
    n::Int                      # Num vars
    m::Int                      # Num cons
    x::Vector{Float64}          # Starting and final solution
    g::Vector{Float64}          # Final constraint values
    mult_g::Vector{Float64}     # lagrange multipliers on constraints
    mult_x_L::Vector{Float64}   # lagrange multipliers on lower bounds
    mult_x_U::Vector{Float64}   # lagrange multipliers on upper bounds
    obj_val::Float64            # Final objective
    status::Cint                # Final status
    # Callbacks
    eval_f::Function
    eval_g::Function
    eval_grad_f::Function
    eval_jac_g::Function
    eval_h::Union{Function,Nothing}
    intermediate::Union{Function,Nothing}
    # User data
    user_model::M
end

Base.unsafe_convert(::Type{Ptr{Cvoid}}, p::IpoptProblem) = p.ipopt_problem

function _Eval_F_CB(
    n::Cint,
    x_ptr::Ptr{Float64},
    x_new::Cint,
    obj_value::Ptr{Float64},
    user_data::Ptr{Cvoid},
)
    prob = unsafe_pointer_to_objref(user_data)::IpoptProblem
    x = unsafe_wrap(Array, x_ptr, Int(n))
    if x_new == Cint(1)
        prob.x .= x
    end
    if isnothing(prob.user_model)
        new_obj = convert(Float64, prob.eval_f(x))::Float64
    else
        new_obj = convert(Float64, prob.eval_f(prob.user_model, x))::Float64
    end
    unsafe_store!(obj_value, new_obj)
    return Cint(1)
end

function _Eval_Grad_F_CB(
    n::Cint,
    x_ptr::Ptr{Float64},
    # A Bool indicating if `x` is a new point. We don't make use of this.
    ::Cint,
    grad_f::Ptr{Float64},
    user_data::Ptr{Cvoid},
)
    prob = unsafe_pointer_to_objref(user_data)::IpoptProblem
    new_grad_f = unsafe_wrap(Array, grad_f, Int(n))
    x = unsafe_wrap(Array, x_ptr, Int(n))
    if isnothing(prob.user_model)
        prob.eval_grad_f(x, new_grad_f)
    else
        prob.eval_grad_f(prob.user_model, x, new_grad_f)
    end
    return Cint(1)
end

function _Eval_G_CB(
    n::Cint,
    x_ptr::Ptr{Float64},
    x_new::Cint,
    m::Cint,
    g_ptr::Ptr{Float64},
    user_data::Ptr{Cvoid},
)
    prob = unsafe_pointer_to_objref(user_data)::IpoptProblem
    new_g = unsafe_wrap(Array, g_ptr, Int(m))
    x = unsafe_wrap(Array, x_ptr, Int(n))
    if x_new == Cint(1)
        prob.x .= x
    end
    if isnothing(prob.user_model)
        prob.eval_g(x, new_g)
    else
        prob.eval_g(prob.user_model, x, new_g)
    end
    return Cint(1)
end

function _Eval_Jac_G_CB(
    n::Cint,
    x_ptr::Ptr{Float64},
    ::Cint,
    ::Cint,
    nele_jac::Cint,
    iRow::Ptr{Cint},
    jCol::Ptr{Cint},
    values_ptr::Ptr{Float64},
    user_data::Ptr{Cvoid},
)
    prob = unsafe_pointer_to_objref(user_data)::IpoptProblem
    x = unsafe_wrap(Array, x_ptr, Int(n))
    rows = unsafe_wrap(Array, iRow, Int(nele_jac))
    cols = unsafe_wrap(Array, jCol, Int(nele_jac))
    if values_ptr == C_NULL
        if isnothing(prob.user_model)
            prob.eval_jac_g(x, rows, cols, nothing)
        else
            prob.eval_jac_g(prob.user_model, x, rows, cols, nothing)
        end
    else
        values = unsafe_wrap(Array, values_ptr, Int(nele_jac))
        if isnothing(prob.user_model)
            prob.eval_jac_g(x, rows, cols, values)
        else
            prob.eval_jac_g(prob.user_model, x, rows, cols, values)
        end
    end
    return Cint(1)
end

function _Eval_H_CB(
    n::Cint,
    x_ptr::Ptr{Float64},
    ::Cint,
    obj_factor::Float64,
    m::Cint,
    lambda_ptr::Ptr{Float64},
    ::Cint,
    nele_hess::Cint,
    iRow::Ptr{Cint},
    jCol::Ptr{Cint},
    values_ptr::Ptr{Float64},
    user_data::Ptr{Cvoid},
)
    prob = unsafe_pointer_to_objref(user_data)::IpoptProblem
    if prob.eval_h === nothing
        # No hessian. Return FALSE for failure.
        return Cint(0)
    end
    x = unsafe_wrap(Array, x_ptr, Int(n))
    lambda = unsafe_wrap(Array, lambda_ptr, Int(m))
    rows = unsafe_wrap(Array, iRow, Int(nele_hess))
    cols = unsafe_wrap(Array, jCol, Int(nele_hess))
    if values_ptr == C_NULL
        if isnothing(prob.user_model)
            prob.eval_h(x, rows, cols, obj_factor, lambda, nothing)
        else
            prob.eval_h(prob.user_model, x, rows, cols, obj_factor, lambda, nothing)
        end
    else
        values = unsafe_wrap(Array, values_ptr, Int(nele_hess))
        if isnothing(prob.user_model)
            prob.eval_h(x, rows, cols, obj_factor, lambda, values)
        else
            prob.eval_h(prob.user_model, x, rows, cols, obj_factor, lambda, values)
        end
    end
    return Cint(1)  # Return TRUE for success.
end

function _Intermediate_CB(
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
)::Cint
    try
        return reenable_sigint() do
            prob = unsafe_pointer_to_objref(user_data)::IpoptProblem
            if isnothing(prob.user_model)
                return prob.intermediate(
                    alg_mod,
                    iter_count,
                    obj_value,
                    inf_pr,
                    inf_du,
                    mu,
                    d_norm,
                    regularization_size,
                    alpha_du,
                    alpha_pr,
                    ls_trials,
                )
            else
                return prob.intermediate(
                    prob.user_model,
                    alg_mod,
                    iter_count,
                    obj_value,
                    inf_pr,
                    inf_du,
                    mu,
                    d_norm,
                    regularization_size,
                    alpha_du,
                    alpha_pr,
                    ls_trials,
                )
            end
        end
    catch err
        if !(err isa InterruptException)
            rethrow(err)
        end
        return false  # optimization should stop
    end
end

function CreateIpoptProblem(
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
    eval_h,
    user_model=nothing,
)
    @assert n == length(x_L) == length(x_U)
    @assert m == length(g_L) == length(g_U)
    eval_f_cb = @cfunction(
        _Eval_F_CB,
        Cint,
        (Cint, Ptr{Float64}, Cint, Ptr{Float64}, Ptr{Cvoid}),
    )
    eval_g_cb = @cfunction(
        _Eval_G_CB,
        Cint,
        (Cint, Ptr{Float64}, Cint, Cint, Ptr{Float64}, Ptr{Cvoid}),
    )
    eval_grad_f_cb = @cfunction(
        _Eval_Grad_F_CB,
        Cint,
        (Cint, Ptr{Float64}, Cint, Ptr{Float64}, Ptr{Cvoid}),
    )
    eval_jac_g_cb = @cfunction(
        _Eval_Jac_G_CB,
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
        _Eval_H_CB,
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
    ipopt_problem = @ccall libipopt.CreateIpoptProblem(
        n::Cint,
        x_L::Ptr{Cdouble},
        x_U::Ptr{Cdouble},
        m::Cint,
        g_L::Ptr{Cdouble},
        g_U::Ptr{Cdouble},
        nele_jac::Cint,
        nele_hess::Cint,
        1::Cint,  # 1 = Fortran style indexing
        eval_f_cb::Ptr{Cvoid},
        eval_g_cb::Ptr{Cvoid},
        eval_grad_f_cb::Ptr{Cvoid},
        eval_jac_g_cb::Ptr{Cvoid},
        eval_h_cb::Ptr{Cvoid},
    )::Ptr{Cvoid}
    if ipopt_problem == C_NULL
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
    prob = IpoptProblem(
        ipopt_problem,
        n,
        m,
        zeros(Float64, n),
        zeros(Float64, m),
        zeros(Float64, m),
        zeros(Float64, n),
        zeros(Float64, n),
        0.0,
        Cint(0),
        eval_f,
        eval_g,
        eval_grad_f,
        eval_jac_g,
        eval_h,
        nothing,
        user_model,
    )
    finalizer(FreeIpoptProblem, prob)
    return prob
end

function FreeIpoptProblem(prob::IpoptProblem)
    @ccall libipopt.FreeIpoptProblem(prob::Ptr{Cvoid})::Cvoid
    return
end

function AddIpoptStrOption(prob::IpoptProblem, keyword::String, value::String)
    if !(isascii(keyword) && isascii(value))
        error("IPOPT: Non ASCII parameters not supported")
    end
    ret = @ccall libipopt.AddIpoptStrOption(
        prob::Ptr{Cvoid},
        keyword::Ptr{UInt8},
        value::Ptr{UInt8},
    )::Bool
    if !ret
        error("IPOPT: Couldn't set option '$keyword' to value '$value'.")
    end
    return
end

function AddIpoptNumOption(prob::IpoptProblem, keyword::String, value::Float64)
    if !isascii(keyword)
        error("IPOPT: Non ASCII parameters not supported")
    end
    ret = @ccall libipopt.AddIpoptNumOption(
        prob::Ptr{Cvoid},
        keyword::Ptr{UInt8},
        value::Cdouble,
    )::Bool
    if !ret
        error("IPOPT: Couldn't set option '$keyword' to value '$value'.")
    end
    return
end

function AddIpoptIntOption(prob::IpoptProblem, keyword::String, value::Integer)
    if !isascii(keyword)
        error("IPOPT: Non ASCII parameters not supported")
    end
    ret = @ccall libipopt.AddIpoptIntOption(
        prob::Ptr{Cvoid},
        keyword::Ptr{UInt8},
        value::Cint,
    )::Bool
    if !ret
        error(
            "IPOPT: Couldn't set option '$keyword' to value '$value'::Int32. " *
            "Note that `Num` options need to be explictly passed as " *
            "`Float64($value)` instead of their integer equivalents.",
        )
    end
    return
end

function OpenIpoptOutputFile(
    prob::IpoptProblem,
    file_name::String,
    print_level::Int,
)
    if !isascii(file_name)
        error("IPOPT: Non ASCII parameters not supported")
    end
    ret = @ccall libipopt.OpenIpoptOutputFile(
        prob::Ptr{Cvoid},
        file_name::Ptr{UInt8},
        print_level::Cint,
    )::Bool
    if !ret
        error("IPOPT: Couldn't open output file.")
    end
    return
end

function SetIpoptProblemScaling(
    prob::IpoptProblem,
    obj_scaling::Float64,
    x_scaling::Union{Ptr{Cvoid},Vector{Float64}},
    g_scaling::Union{Ptr{Cvoid},Vector{Float64}},
)
    ret = @ccall libipopt.SetIpoptProblemScaling(
        prob::Ptr{Cvoid},
        obj_scaling::Cdouble,
        x_scaling::Ptr{Cdouble},
        g_scaling::Ptr{Cdouble},
    )::Bool
    @assert ret  # The C++ code has `return true`
    return
end

function SetIntermediateCallback(prob::IpoptProblem, intermediate::Function)
    intermediate_cb = @cfunction(
        _Intermediate_CB,
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
    ret = @ccall libipopt.SetIntermediateCallback(
        prob::Ptr{Cvoid},
        intermediate_cb::Ptr{Cvoid},
    )::Bool
    @assert ret  # The C++ code has `return true`
    prob.intermediate = intermediate
    return
end

function IpoptSolve(prob::IpoptProblem)
    p_objval = Ref{Cdouble}(0.0)
    disable_sigint() do
        prob.status = @ccall libipopt.IpoptSolve(
            prob::Ptr{Cvoid},
            prob.x::Ptr{Cdouble},
            prob.g::Ptr{Cdouble},
            p_objval::Ptr{Cdouble},
            prob.mult_g::Ptr{Cdouble},
            prob.mult_x_L::Ptr{Cdouble},
            prob.mult_x_U::Ptr{Cdouble},
            pointer_from_objref(prob)::Ptr{Cvoid},
        )::Cint
        return
    end
    prob.obj_val = p_objval[]
    return prob.status
end

function GetIpoptCurrentIterate(
    prob::IpoptProblem,
    scaled::Bool,
    n::Integer,
    x::Union{Ptr{Cvoid},Vector{Float64}},
    z_L::Union{Ptr{Cvoid},Vector{Float64}},
    z_U::Union{Ptr{Cvoid},Vector{Float64}},
    m::Integer,
    g::Union{Ptr{Cvoid},Vector{Float64}},
    lambda::Union{Ptr{Cvoid},Vector{Float64}},
)
    ret = @ccall libipopt.GetIpoptCurrentIterate(
        prob::Ptr{Cvoid},
        scaled::Bool,
        n::Cint,
        x::Ptr{Cdouble},
        z_L::Ptr{Cdouble},
        z_U::Ptr{Cdouble},
        m::Cint,
        g::Ptr{Cdouble},
        lambda::Ptr{Cdouble},
    )::Bool
    if !ret
        error("IPOPT: Something went wrong getting the current iterate.")
    end
    return
end

function GetIpoptCurrentViolations(
    prob::IpoptProblem,
    scaled::Bool,
    n::Integer,
    x_L_violation::Union{Ptr{Cvoid},Vector{Float64}},
    x_U_violation::Union{Ptr{Cvoid},Vector{Float64}},
    compl_x_L::Union{Ptr{Cvoid},Vector{Float64}},
    compl_x_U::Union{Ptr{Cvoid},Vector{Float64}},
    grad_lag_x::Union{Ptr{Cvoid},Vector{Float64}},
    m::Integer,
    nlp_constraint_violation::Union{Ptr{Cvoid},Vector{Float64}},
    compl_g::Union{Ptr{Cvoid},Vector{Float64}},
)
    ret = @ccall libipopt.GetIpoptCurrentViolations(
        prob::Ptr{Cvoid},
        scaled::Bool,
        n::Cint,
        x_L_violation::Ptr{Cdouble},
        x_U_violation::Ptr{Cdouble},
        compl_x_L::Ptr{Cdouble},
        compl_x_U::Ptr{Cdouble},
        grad_lag_x::Ptr{Cdouble},
        m::Cint,
        nlp_constraint_violation::Ptr{Cdouble},
        compl_g::Ptr{Cdouble},
    )::Bool
    if !ret
        error("IPOPT: Something went wrong getting the current violations.")
    end
    return
end

function GetIpoptVersion()
    major, minor, patch = Ref{Cint}(), Ref{Cint}(), Ref{Cint}()
    @ccall libipopt.GetIpoptVersion(
        major::Ptr{Cint},
        minor::Ptr{Cint},
        patch::Ptr{Cint},
    )::Cvoid
    return VersionNumber(major[], minor[], patch[])
end

# https://github.com/coin-or/Ipopt/blob/8f2b8efcd53d93518984597808db05dce43e348f/src/Interfaces/IpReturnCodes_inc.h#L13-L38
#!format:off
@enum(
   ApplicationReturnStatus,
   Solve_Succeeded                    = 0,
   Solved_To_Acceptable_Level         = 1,
   Infeasible_Problem_Detected        = 2,
   Search_Direction_Becomes_Too_Small = 3,
   Diverging_Iterates                 = 4,
   User_Requested_Stop                = 5,
   Feasible_Point_Found               = 6,

   Maximum_Iterations_Exceeded        = -1,
   Restoration_Failed                 = -2,
   Error_In_Step_Computation          = -3,
   Maximum_CpuTime_Exceeded           = -4,
   Maximum_WallTime_Exceeded          = -5,   # @since 3.14.0

   Not_Enough_Degrees_Of_Freedom      = -10,
   Invalid_Problem_Definition         = -11,
   Invalid_Option                     = -12,
   Invalid_Number_Detected            = -13,

   Unrecoverable_Exception            = -100,
   NonIpopt_Exception_Thrown          = -101,
   Insufficient_Memory                = -102,
   Internal_Error                     = -199
)
#!format:on
