# Copyright (c) 2013: Iain Dunning, Miles Lubin, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

mutable struct IpoptProblem
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
end

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
    new_obj = convert(Float64, prob.eval_f(x))::Float64
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
    prob.eval_grad_f(x, new_grad_f)
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
    prob.eval_g(x, new_g)
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
        prob.eval_jac_g(x, rows, cols, nothing)
    else
        values = unsafe_wrap(Array, values_ptr, Int(nele_jac))
        prob.eval_jac_g(x, rows, cols, values)
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
        prob.eval_h(x, rows, cols, obj_factor, lambda, nothing)
    else
        values = unsafe_wrap(Array, values_ptr, Int(nele_hess))
        prob.eval_h(x, rows, cols, obj_factor, lambda, values)
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
)
    prob = unsafe_pointer_to_objref(user_data)::IpoptProblem
    ret = prob.intermediate(
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
    # Return TRUE if the optimization should continue.
    return ret ? Cint(1) : Cint(0)
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
    ipopt_problem = ccall(
        (:CreateIpoptProblem, libipopt),
        Ptr{Cvoid},
        (
            Cint,
            Ptr{Float64},
            Ptr{Float64},
            Cint,
            Ptr{Float64},
            Ptr{Float64},
            Cint,
            Cint,
            Cint,
            Ptr{Cvoid},
            Ptr{Cvoid},
            Ptr{Cvoid},
            Ptr{Cvoid},
            Ptr{Cvoid},
        ),
        n,
        x_L,
        x_U,
        m,
        g_L,
        g_U,
        nele_jac,
        nele_hess,
        1,  # 1 = Fortran style indexing
        eval_f_cb,
        eval_g_cb,
        eval_grad_f_cb,
        eval_jac_g_cb,
        eval_h_cb,
    )
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
        0,
        eval_f,
        eval_g,
        eval_grad_f,
        eval_jac_g,
        eval_h,
        nothing,
    )
    finalizer(FreeIpoptProblem, prob)
    return prob
end

function FreeIpoptProblem(prob::IpoptProblem)
    ccall(
        (:FreeIpoptProblem, libipopt),
        Cvoid,
        (Ptr{Cvoid},),
        prob.ipopt_problem,
    )
    return
end

function AddIpoptStrOption(prob::IpoptProblem, keyword::String, value::String)
    if !(isascii(keyword) && isascii(value))
        error("IPOPT: Non ASCII parameters not supported")
    end
    ret = ccall(
        (:AddIpoptStrOption, libipopt),
        Bool,
        (Ptr{Cvoid}, Ptr{UInt8}, Ptr{UInt8}),
        prob.ipopt_problem,
        keyword,
        value,
    )
    if !ret
        error("IPOPT: Couldn't set option '$keyword' to value '$value'.")
    end
    return
end

function AddIpoptNumOption(prob::IpoptProblem, keyword::String, value::Float64)
    if !isascii(keyword)
        error("IPOPT: Non ASCII parameters not supported")
    end
    ret = ccall(
        (:AddIpoptNumOption, libipopt),
        Bool,
        (Ptr{Cvoid}, Ptr{UInt8}, Float64),
        prob.ipopt_problem,
        keyword,
        value,
    )
    if !ret
        error("IPOPT: Couldn't set option '$keyword' to value '$value'.")
    end
    return
end

function AddIpoptIntOption(prob::IpoptProblem, keyword::String, value::Integer)
    if !isascii(keyword)
        error("IPOPT: Non ASCII parameters not supported")
    end
    ret = ccall(
        (:AddIpoptIntOption, libipopt),
        Bool,
        (Ptr{Cvoid}, Ptr{UInt8}, Cint),
        prob.ipopt_problem,
        keyword,
        value,
    )
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
    ret = ccall(
        (:OpenIpoptOutputFile, libipopt),
        Bool,
        (Ptr{Cvoid}, Ptr{UInt8}, Cint),
        prob.ipopt_problem,
        file_name,
        print_level,
    )
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
    ret = ccall(
        (:SetIpoptProblemScaling, libipopt),
        Bool,
        (Ptr{Cvoid}, Float64, Ptr{Float64}, Ptr{Float64}),
        prob.ipopt_problem,
        obj_scaling,
        x_scaling,
        g_scaling,
    )
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
    ret = ccall(
        (:SetIntermediateCallback, libipopt),
        Bool,
        (Ptr{Cvoid}, Ptr{Cvoid}),
        prob.ipopt_problem,
        intermediate_cb,
    )
    @assert ret  # The C++ code has `return true`
    prob.intermediate = intermediate
    return
end

function IpoptSolve(prob::IpoptProblem)
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
            Ptr{Cvoid},
        ),
        prob.ipopt_problem,
        prob.x,
        prob.g,
        final_objval,
        prob.mult_g,
        prob.mult_x_L,
        prob.mult_x_U,
        pointer_from_objref(prob),
    )
    prob.obj_val = final_objval[]
    prob.status = ret
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
    ret = ccall(
        (:GetIpoptCurrentIterate, libipopt),
        Bool,
        (
            Ptr{Cvoid},
            Cint,
            Cint,
            Ptr{Float64},
            Ptr{Float64},
            Ptr{Float64},
            Cint,
            Ptr{Float64},
            Ptr{Float64},
        ),
        prob.ipopt_problem,
        scaled,
        n,
        x,
        z_L,
        z_U,
        m,
        g,
        lambda,
    )
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
    ret = ccall(
        (:GetIpoptCurrentViolations, libipopt),
        Bool,
        (
            Ptr{Cvoid},
            Cint,
            Cint,
            Ptr{Float64},
            Ptr{Float64},
            Ptr{Float64},
            Ptr{Float64},
            Ptr{Float64},
            Cint,
            Ptr{Float64},
            Ptr{Float64},
        ),
        prob.ipopt_problem,
        scaled,
        n,
        x_L_violation,
        x_U_violation,
        compl_x_L,
        compl_x_U,
        grad_lag_x,
        m,
        nlp_constraint_violation,
        compl_g,
    )
    if !ret
        error("IPOPT: Something went wrong getting the current violations.")
    end
    return
end
