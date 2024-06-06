# Copyright (c) 2013: Iain Dunning, Miles Lubin, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

include("utils.jl")

const _PARAMETER_OFFSET = 0x00f0000000000000

_is_parameter(x::MOI.VariableIndex) = x.value >= _PARAMETER_OFFSET

_is_parameter(term::MOI.ScalarAffineTerm) = _is_parameter(term.variable)

function _is_parameter(term::MOI.ScalarQuadraticTerm)
    return _is_parameter(term.variable_1) || _is_parameter(term.variable_2)
end

"""
    Optimizer()

Create a new Ipopt optimizer.
"""
mutable struct Optimizer <: MOI.AbstractOptimizer
    inner::Union{Nothing,IpoptProblem}
    name::String
    invalid_model::Bool
    silent::Bool
    options::Dict{String,Any}
    solve_time::Float64
    sense::MOI.OptimizationSense

    parameters::Dict{MOI.VariableIndex,MOI.Nonlinear.ParameterIndex}
    variables::MOI.Utilities.VariablesContainer{Float64}
    list_of_variable_indices::Vector{MOI.VariableIndex}
    variable_primal_start::Vector{Union{Nothing,Float64}}
    mult_x_L::Vector{Union{Nothing,Float64}}
    mult_x_U::Vector{Union{Nothing,Float64}}

    nlp_data::MOI.NLPBlockData
    nlp_dual_start::Union{Nothing,Vector{Float64}}

    qp_data::QPBlockData{Float64}
    nlp_model::Union{Nothing,MOI.Nonlinear.Model}
    callback::Union{Nothing,Function}
    barrier_iterations::Int
    ad_backend::MOI.Nonlinear.AbstractAutomaticDifferentiation

    function Optimizer()
        return new(
            nothing,
            "",
            false,
            false,
            Dict{String,Any}(),
            NaN,
            MOI.FEASIBILITY_SENSE,
            Dict{MOI.VariableIndex,Float64}(),
            MOI.Utilities.VariablesContainer{Float64}(),
            MOI.VariableIndex[],
            Union{Nothing,Float64}[],
            Union{Nothing,Float64}[],
            Union{Nothing,Float64}[],
            MOI.NLPBlockData([], _EmptyNLPEvaluator(), false),
            nothing,
            QPBlockData{Float64}(),
            nothing,
            nothing,
            0,
            MOI.Nonlinear.SparseReverseMode(),
        )
    end
end

const _SETS = Union{
    MOI.GreaterThan{Float64},
    MOI.LessThan{Float64},
    MOI.EqualTo{Float64},
    MOI.Interval{Float64},
}

const _FUNCTIONS = Union{
    MOI.ScalarAffineFunction{Float64},
    MOI.ScalarQuadraticFunction{Float64},
    MOI.ScalarNonlinearFunction,
}

MOI.get(::Optimizer, ::MOI.SolverVersion) = "3.14.4"

### _EmptyNLPEvaluator

struct _EmptyNLPEvaluator <: MOI.AbstractNLPEvaluator end

MOI.features_available(::_EmptyNLPEvaluator) = [:Grad, :Jac, :Hess]
MOI.initialize(::_EmptyNLPEvaluator, ::Any) = nothing
MOI.eval_constraint(::_EmptyNLPEvaluator, g, x) = nothing
MOI.jacobian_structure(::_EmptyNLPEvaluator) = Tuple{Int64,Int64}[]
MOI.hessian_lagrangian_structure(::_EmptyNLPEvaluator) = Tuple{Int64,Int64}[]
MOI.eval_constraint_jacobian(::_EmptyNLPEvaluator, J, x) = nothing
MOI.eval_hessian_lagrangian(::_EmptyNLPEvaluator, H, x, σ, μ) = nothing

function MOI.empty!(model::Optimizer)
    model.inner = nothing
    model.invalid_model = false
    model.sense = MOI.FEASIBILITY_SENSE
    empty!(model.parameters)
    MOI.empty!(model.variables)
    empty!(model.list_of_variable_indices)
    empty!(model.variable_primal_start)
    empty!(model.mult_x_L)
    empty!(model.mult_x_U)
    model.nlp_data = MOI.NLPBlockData([], _EmptyNLPEvaluator(), false)
    model.nlp_dual_start = nothing
    model.nlp_model = nothing
    model.qp_data = QPBlockData{Float64}()
    model.callback = nothing
    model.barrier_iterations = 0
    return
end

function MOI.is_empty(model::Optimizer)
    return MOI.is_empty(model.variables) &&
           isempty(model.variable_primal_start) &&
           isempty(model.mult_x_L) &&
           isempty(model.mult_x_U) &&
           model.nlp_data.evaluator isa _EmptyNLPEvaluator &&
           model.sense == MOI.FEASIBILITY_SENSE
end

MOI.supports_incremental_interface(::Optimizer) = true

function MOI.copy_to(model::Optimizer, src::MOI.ModelLike)
    return MOI.Utilities.default_copy_to(model, src)
end

MOI.get(::Optimizer, ::MOI.SolverName) = "Ipopt"

function MOI.supports_add_constrained_variable(
    ::Optimizer,
    ::Type{MOI.Parameter{Float64}},
)
    return true
end

function _init_nlp_model(model)
    if model.nlp_model === nothing
        if !(model.nlp_data.evaluator isa _EmptyNLPEvaluator)
            error("Cannot mix the new and legacy nonlinear APIs")
        end
        model.nlp_model = MOI.Nonlinear.Model()
    end
    return
end

function MOI.add_constrained_variable(
    model::Optimizer,
    set::MOI.Parameter{Float64},
)
    model.inner = nothing
    _init_nlp_model(model)
    p = MOI.VariableIndex(_PARAMETER_OFFSET + length(model.parameters))
    push!(model.list_of_variable_indices, p)
    model.parameters[p] =
        MOI.Nonlinear.add_parameter(model.nlp_model, set.value)
    ci = MOI.ConstraintIndex{MOI.VariableIndex,typeof(set)}(p.value)
    return p, ci
end

function MOI.is_valid(
    model::Optimizer,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{Float64}},
)
    p = MOI.VariableIndex(ci.value)
    return haskey(model.parameters, p)
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintSet,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{Float64}},
    set::MOI.Parameter{Float64},
)
    p = model.parameters[MOI.VariableIndex(ci.value)]
    model.nlp_model[p] = set.value
    return
end

_replace_parameters(model::Optimizer, f) = f

function _replace_parameters(model::Optimizer, f::MOI.VariableIndex)
    if _is_parameter(f)
        return model.parameters[f]
    end
    return f
end

function _replace_parameters(model::Optimizer, f::MOI.ScalarAffineFunction)
    if any(_is_parameter, f.terms)
        g = convert(MOI.ScalarNonlinearFunction, f)
        return _replace_parameters(model, g)
    end
    return f
end

function _replace_parameters(model::Optimizer, f::MOI.ScalarQuadraticFunction)
    if any(_is_parameter, f.affine_terms) ||
       any(_is_parameter, f.quadratic_terms)
        g = convert(MOI.ScalarNonlinearFunction, f)
        return _replace_parameters(model, g)
    end
    return f
end

function _replace_parameters(model::Optimizer, f::MOI.ScalarNonlinearFunction)
    for (i, arg) in enumerate(f.args)
        f.args[i] = _replace_parameters(model, arg)
    end
    return f
end

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{<:Union{MOI.VariableIndex,_FUNCTIONS}},
    ::Type{<:_SETS},
)
    return true
end

### MOI.ListOfConstraintTypesPresent

function MOI.get(model::Optimizer, attr::MOI.ListOfConstraintTypesPresent)
    ret = MOI.get(model.variables, attr)
    append!(ret, MOI.get(model.qp_data, attr))
    return ret
end

### MOI.Name

MOI.supports(::Optimizer, ::MOI.Name) = true

function MOI.set(model::Optimizer, ::MOI.Name, value::String)
    model.name = value
    return
end

MOI.get(model::Optimizer, ::MOI.Name) = model.name

### MOI.Silent

MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.set(model::Optimizer, ::MOI.Silent, value)
    model.silent = value
    return
end

MOI.get(model::Optimizer, ::MOI.Silent) = model.silent

### MOI.TimeLimitSec

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true

function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, value::Real)
    MOI.set(model, MOI.RawOptimizerAttribute("max_wall_time"), Float64(value))
    return
end

function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, ::Nothing)
    delete!(model.options, "max_wall_time")
    return
end

function MOI.get(model::Optimizer, ::MOI.TimeLimitSec)
    return get(model.options, "max_wall_time", nothing)
end

### MOI.RawOptimizerAttribute

MOI.supports(::Optimizer, ::MOI.RawOptimizerAttribute) = true

function MOI.set(model::Optimizer, p::MOI.RawOptimizerAttribute, value)
    model.options[p.name] = value
    # No need to reset model.inner because this gets handled in optimize!.
    return
end

function MOI.get(model::Optimizer, p::MOI.RawOptimizerAttribute)
    if !haskey(model.options, p.name)
        error("RawParameter with name $(p.name) is not set.")
    end
    return model.options[p.name]
end

### Variables

"""
    column(x::MOI.VariableIndex)

Return the column associated with a variable.
"""
column(x::MOI.VariableIndex) = x.value

function MOI.add_variable(model::Optimizer)
    push!(model.variable_primal_start, nothing)
    push!(model.mult_x_L, nothing)
    push!(model.mult_x_U, nothing)
    model.inner = nothing
    x = MOI.add_variable(model.variables)
    push!(model.list_of_variable_indices, x)
    return x
end

function MOI.is_valid(model::Optimizer, x::MOI.VariableIndex)
    if _is_parameter(x)
        return haskey(model.parameters, x)
    end
    return MOI.is_valid(model.variables, x)
end

function MOI.get(model::Optimizer, ::MOI.ListOfVariableIndices)
    return model.list_of_variable_indices
end

function MOI.get(model::Optimizer, ::MOI.NumberOfVariables)
    return length(model.list_of_variable_indices)
end

function MOI.is_valid(
    model::Optimizer,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,<:_SETS},
)
    return MOI.is_valid(model.variables, ci)
end

function MOI.get(
    model::Optimizer,
    attr::Union{
        MOI.NumberOfConstraints{MOI.VariableIndex,<:_SETS},
        MOI.ListOfConstraintIndices{MOI.VariableIndex,<:_SETS},
    },
)
    return MOI.get(model.variables, attr)
end

function MOI.get(
    model::Optimizer,
    attr::Union{MOI.ConstraintFunction,MOI.ConstraintSet},
    c::MOI.ConstraintIndex{MOI.VariableIndex,<:_SETS},
)
    return MOI.get(model.variables, attr, c)
end

function MOI.add_constraint(model::Optimizer, x::MOI.VariableIndex, set::_SETS)
    index = MOI.add_constraint(model.variables, x, set)
    model.inner = nothing
    return index
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintSet,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,S},
    set::S,
) where {S<:_SETS}
    MOI.set(model.variables, MOI.ConstraintSet(), ci, set)
    model.inner = nothing
    return
end

function MOI.delete(
    model::Optimizer,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,<:_SETS},
)
    MOI.delete(model.variables, ci)
    model.inner = nothing
    return
end

### ScalarAffineFunction and ScalarQuadraticFunction constraints

function MOI.is_valid(
    model::Optimizer,
    ci::MOI.ConstraintIndex{<:_FUNCTIONS,<:_SETS},
)
    return MOI.is_valid(model.qp_data, ci)
end

function MOI.add_constraint(model::Optimizer, func::_FUNCTIONS, set::_SETS)
    index = MOI.add_constraint(model.qp_data, func, set)
    model.inner = nothing
    return index
end

function MOI.get(
    model::Optimizer,
    attr::Union{MOI.NumberOfConstraints{F,S},MOI.ListOfConstraintIndices{F,S}},
) where {F<:_FUNCTIONS,S<:_SETS}
    return MOI.get(model.qp_data, attr)
end

function MOI.get(
    model::Optimizer,
    attr::Union{
        MOI.ConstraintFunction,
        MOI.ConstraintSet,
        MOI.ConstraintDualStart,
    },
    c::MOI.ConstraintIndex{F,S},
) where {F<:_FUNCTIONS,S<:_SETS}
    return MOI.get(model.qp_data, attr, c)
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintSet,
    ci::MOI.ConstraintIndex{F,S},
    set::S,
) where {F<:_FUNCTIONS,S<:_SETS}
    MOI.set(model.qp_data, MOI.ConstraintSet(), ci, set)
    model.inner = nothing
    return
end

function MOI.supports(
    ::Optimizer,
    ::MOI.ConstraintDualStart,
    ::Type{MOI.ConstraintIndex{F,S}},
) where {F<:_FUNCTIONS,S<:_SETS}
    return true
end

function MOI.set(
    model::Optimizer,
    attr::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{F,S},
    value::Union{Real,Nothing},
) where {F<:_FUNCTIONS,S<:_SETS}
    MOI.throw_if_not_valid(model, ci)
    MOI.set(model.qp_data, attr, ci, value)
    # No need to reset model.inner, because this gets handled in optimize!.
    return
end

### ScalarNonlinearFunction

function MOI.is_valid(
    model::Optimizer,
    ci::MOI.ConstraintIndex{MOI.ScalarNonlinearFunction,<:_SETS},
)
    if model.nlp_model === nothing
        return false
    end
    index = MOI.Nonlinear.ConstraintIndex(ci.value)
    return MOI.is_valid(model.nlp_model, index)
end

function MOI.add_constraint(
    model::Optimizer,
    f::MOI.ScalarNonlinearFunction,
    s::_SETS,
)
    _init_nlp_model(model)
    if !isempty(model.parameters)
        _replace_parameters(model, f)
    end
    index = MOI.Nonlinear.add_constraint(model.nlp_model, f, s)
    model.inner = nothing
    return MOI.ConstraintIndex{typeof(f),typeof(s)}(index.value)
end

function MOI.set(
    model::Optimizer,
    attr::MOI.ObjectiveFunction{MOI.ScalarNonlinearFunction},
    func::MOI.ScalarNonlinearFunction,
)
    _init_nlp_model(model)
    if !isempty(model.parameters)
        _replace_parameters(model, func)
    end
    MOI.Nonlinear.set_objective(model.nlp_model, func)
    model.inner = nothing
    return
end

### UserDefinedFunction

MOI.supports(model::Optimizer, ::MOI.UserDefinedFunction) = true

function MOI.set(model::Optimizer, attr::MOI.UserDefinedFunction, args)
    _init_nlp_model(model)
    MOI.Nonlinear.register_operator(
        model.nlp_model,
        attr.name,
        attr.arity,
        args...,
    )
    return
end

### ListOfSupportedNonlinearOperators

function MOI.get(model::Optimizer, attr::MOI.ListOfSupportedNonlinearOperators)
    _init_nlp_model(model)
    return MOI.get(model.nlp_model, attr)
end

### MOI.VariablePrimalStart

function MOI.supports(
    ::Optimizer,
    ::MOI.VariablePrimalStart,
    ::Type{MOI.VariableIndex},
)
    return true
end

function MOI.set(
    model::Optimizer,
    ::MOI.VariablePrimalStart,
    vi::MOI.VariableIndex,
    value::Union{Real,Nothing},
)
    if _is_parameter(vi)
        return  # Do nothing
    end
    MOI.throw_if_not_valid(model, vi)
    model.variable_primal_start[column(vi)] = value
    # No need to reset model.inner, because this gets handled in optimize!.
    return
end

### MOI.ConstraintDualStart

_dual_start(::Optimizer, ::Nothing, ::Int = 1) = 0.0

function _dual_start(model::Optimizer, value::Real, scale::Int = 1)
    return _dual_multiplier(model) * value * scale
end

function MOI.supports(
    ::Optimizer,
    ::MOI.ConstraintDualStart,
    ::Type{MOI.ConstraintIndex{MOI.VariableIndex,S}},
) where {S<:_SETS}
    return true
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.GreaterThan{Float64}},
    value::Union{Real,Nothing},
)
    MOI.throw_if_not_valid(model, ci)
    model.mult_x_L[ci.value] = value
    # No need to reset model.inner, because this gets handled in optimize!.
    return
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.GreaterThan{Float64}},
)
    MOI.throw_if_not_valid(model, ci)
    return model.mult_x_L[ci.value]
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.LessThan{Float64}},
    value::Union{Real,Nothing},
)
    MOI.throw_if_not_valid(model, ci)
    model.mult_x_U[ci.value] = value
    # No need to reset model.inner, because this gets handled in optimize!.
    return
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.LessThan{Float64}},
)
    MOI.throw_if_not_valid(model, ci)
    return model.mult_x_U[ci.value]
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.EqualTo{Float64}},
    value::Union{Real,Nothing},
)
    MOI.throw_if_not_valid(model, ci)
    if value === nothing
        model.mult_x_L[ci.value] = nothing
        model.mult_x_U[ci.value] = nothing
    elseif value >= 0.0
        model.mult_x_L[ci.value] = value
        model.mult_x_U[ci.value] = 0.0
    else
        model.mult_x_L[ci.value] = 0.0
        model.mult_x_U[ci.value] = value
    end
    # No need to reset model.inner, because this gets handled in optimize!.
    return
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.EqualTo{Float64}},
)
    MOI.throw_if_not_valid(model, ci)
    l = model.mult_x_L[ci.value]
    u = model.mult_x_U[ci.value]
    return (l === u === nothing) ? nothing : (l + u)
end

### MOI.NLPBlockDualStart

MOI.supports(::Optimizer, ::MOI.NLPBlockDualStart) = true

function MOI.set(
    model::Optimizer,
    ::MOI.NLPBlockDualStart,
    values::Union{Nothing,Vector},
)
    model.nlp_dual_start = values
    # No need to reset model.inner, because this gets handled in optimize!.
    return
end

MOI.get(model::Optimizer, ::MOI.NLPBlockDualStart) = model.nlp_dual_start

### MOI.NLPBlock

MOI.supports(::Optimizer, ::MOI.NLPBlock) = true

# This may also be set by `optimize!` and contain the block created from
# ScalarNonlinearFunction
MOI.get(model::Optimizer, ::MOI.NLPBlock) = model.nlp_data

function MOI.set(model::Optimizer, ::MOI.NLPBlock, nlp_data::MOI.NLPBlockData)
    if model.nlp_model !== nothing
        error("Cannot mix the new and legacy nonlinear APIs")
    end
    model.nlp_data = nlp_data
    model.inner = nothing
    return
end

### ObjectiveSense

MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true

function MOI.set(
    model::Optimizer,
    ::MOI.ObjectiveSense,
    sense::MOI.OptimizationSense,
)
    model.sense = sense
    model.inner = nothing
    return
end

MOI.get(model::Optimizer, ::MOI.ObjectiveSense) = model.sense

### ObjectiveFunction

function MOI.get(
    model::Optimizer,
    attr::Union{MOI.ObjectiveFunctionType,MOI.ObjectiveFunction},
)
    return MOI.get(model.qp_data, attr)
end

function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{<:Union{MOI.VariableIndex,<:_FUNCTIONS}},
)
    return true
end

function MOI.set(
    model::Optimizer,
    attr::MOI.ObjectiveFunction{F},
    func::F,
) where {F<:Union{MOI.VariableIndex,<:_FUNCTIONS}}
    MOI.set(model.qp_data, attr, func)
    if model.nlp_model !== nothing
        MOI.Nonlinear.set_objective(model.nlp_model, nothing)
    end
    model.inner = nothing
    return
end

### Eval_F_CB

function MOI.eval_objective(model::Optimizer, x)
    # TODO(odow): FEASIBILITY_SENSE could produce confusing solver output if
    # a nonzero objective is set.
    if model.sense == MOI.FEASIBILITY_SENSE
        return 0.0
    elseif model.nlp_data.has_objective
        return MOI.eval_objective(model.nlp_data.evaluator, x)
    end
    return MOI.eval_objective(model.qp_data, x)
end

### Eval_Grad_F_CB

function MOI.eval_objective_gradient(model::Optimizer, grad, x)
    if model.sense == MOI.FEASIBILITY_SENSE
        grad .= zero(eltype(grad))
    elseif model.nlp_data.has_objective
        MOI.eval_objective_gradient(model.nlp_data.evaluator, grad, x)
    else
        MOI.eval_objective_gradient(model.qp_data, grad, x)
    end
    return
end

### Eval_G_CB

function MOI.eval_constraint(model::Optimizer, g, x)
    MOI.eval_constraint(model.qp_data, g, x)
    g_nlp = view(g, (length(model.qp_data)+1):length(g))
    MOI.eval_constraint(model.nlp_data.evaluator, g_nlp, x)
    return
end

### Eval_Jac_G_CB

function MOI.jacobian_structure(model::Optimizer)
    J = MOI.jacobian_structure(model.qp_data)
    offset = length(model.qp_data)
    if length(model.nlp_data.constraint_bounds) > 0
        for (row, col) in MOI.jacobian_structure(model.nlp_data.evaluator)
            push!(J, (row + offset, col))
        end
    end
    return J
end

function MOI.eval_constraint_jacobian(model::Optimizer, values, x)
    offset = MOI.eval_constraint_jacobian(model.qp_data, values, x)
    nlp_values = view(values, offset:length(values))
    MOI.eval_constraint_jacobian(model.nlp_data.evaluator, nlp_values, x)
    return
end

### Eval_H_CB

function MOI.hessian_lagrangian_structure(model::Optimizer)
    H = MOI.hessian_lagrangian_structure(model.qp_data)
    append!(H, MOI.hessian_lagrangian_structure(model.nlp_data.evaluator))
    return H
end

function MOI.eval_hessian_lagrangian(model::Optimizer, H, x, σ, μ)
    offset = MOI.eval_hessian_lagrangian(model.qp_data, H, x, σ, μ)
    H_nlp = view(H, offset:length(H))
    μ_nlp = view(μ, (length(model.qp_data)+1):length(μ))
    MOI.eval_hessian_lagrangian(model.nlp_data.evaluator, H_nlp, x, σ, μ_nlp)
    return
end

### MOI.AutomaticDifferentiationBackend

MOI.supports(::Optimizer, ::MOI.AutomaticDifferentiationBackend) = true

function MOI.get(model::Optimizer, ::MOI.AutomaticDifferentiationBackend)
    return model.ad_backend
end

function MOI.set(
    model::Optimizer,
    ::MOI.AutomaticDifferentiationBackend,
    backend::MOI.Nonlinear.AbstractAutomaticDifferentiation,
)
    model.ad_backend = backend
    return
end

### MOI.optimize!

function _setup_model(model::Optimizer)
    vars = MOI.get(model.variables, MOI.ListOfVariableIndices())
    if isempty(vars)
        # Don't attempt to create a problem because Ipopt will error.
        model.invalid_model = true
        return
    end
    if model.nlp_model !== nothing
        model.nlp_data = MOI.NLPBlockData(
            MOI.Nonlinear.Evaluator(model.nlp_model, model.ad_backend, vars),
        )
    end
    has_quadratic_constraints =
        any(isequal(_kFunctionTypeScalarQuadratic), model.qp_data.function_type)
    has_nlp_constraints = !isempty(model.nlp_data.constraint_bounds)
    has_hessian = :Hess in MOI.features_available(model.nlp_data.evaluator)
    init_feat = [:Grad]
    if has_hessian
        push!(init_feat, :Hess)
    end
    if has_nlp_constraints
        push!(init_feat, :Jac)
    end
    MOI.initialize(model.nlp_data.evaluator, init_feat)
    jacobian_sparsity = MOI.jacobian_structure(model)
    hessian_sparsity = if has_hessian
        MOI.hessian_lagrangian_structure(model)
    else
        Tuple{Int,Int}[]
    end
    eval_f_cb(x) = MOI.eval_objective(model, x)
    eval_grad_f_cb(x, grad_f) = MOI.eval_objective_gradient(model, grad_f, x)
    eval_g_cb(x, g) = MOI.eval_constraint(model, g, x)
    function eval_jac_g_cb(x, rows, cols, values)
        if values === nothing
            for i in 1:length(jacobian_sparsity)
                rows[i], cols[i] = jacobian_sparsity[i]
            end
        else
            MOI.eval_constraint_jacobian(model, values, x)
        end
        return
    end
    function eval_h_cb(x, rows, cols, obj_factor, lambda, values)
        if values === nothing
            for i in 1:length(hessian_sparsity)
                rows[i], cols[i] = hessian_sparsity[i]
            end
        else
            MOI.eval_hessian_lagrangian(model, values, x, obj_factor, lambda)
        end
        return
    end
    g_L, g_U = copy(model.qp_data.g_L), copy(model.qp_data.g_U)
    for bound in model.nlp_data.constraint_bounds
        push!(g_L, bound.lower)
        push!(g_U, bound.upper)
    end
    model.inner = CreateIpoptProblem(
        length(vars),
        model.variables.lower,
        model.variables.upper,
        length(g_L),
        g_L,
        g_U,
        length(jacobian_sparsity),
        length(hessian_sparsity),
        eval_f_cb,
        eval_g_cb,
        eval_grad_f_cb,
        eval_jac_g_cb,
        has_hessian ? eval_h_cb : nothing,
    )
    if model.sense == MOI.MIN_SENSE
        AddIpoptNumOption(model.inner, "obj_scaling_factor", 1.0)
    elseif model.sense == MOI.MAX_SENSE
        AddIpoptNumOption(model.inner, "obj_scaling_factor", -1.0)
    end
    # Ipopt crashes by default if NaN/Inf values are returned from the
    # evaluation callbacks. This option tells Ipopt to explicitly check for them
    # and return Invalid_Number_Detected instead. This setting may result in a
    # minor performance loss and can be overwritten by specifying
    # check_derivatives_for_naninf="no".
    AddIpoptStrOption(model.inner, "check_derivatives_for_naninf", "yes")
    if !has_hessian
        AddIpoptStrOption(
            model.inner,
            "hessian_approximation",
            "limited-memory",
        )
    end
    if !has_nlp_constraints && !has_quadratic_constraints
        AddIpoptStrOption(model.inner, "jac_c_constant", "yes")
        AddIpoptStrOption(model.inner, "jac_d_constant", "yes")
        if !model.nlp_data.has_objective
            # We turn on this option if all constraints are linear and the
            # objective is linear or quadratic. From the documentation, it's
            # unclear if it may also apply if the constraints are at most
            # quadratic.
            AddIpoptStrOption(model.inner, "hessian_constant", "yes")
        end
    end
    return
end

function copy_parameters(model::Optimizer)
    if model.nlp_model === nothing
        return
    end
    empty!(model.qp_data.parameters)
    for (p, index) in model.parameters
        model.qp_data.parameters[p.value] = model.nlp_model[index]
    end
    return
end

function MOI.optimize!(model::Optimizer)
    start_time = time()
    if model.inner === nothing
        _setup_model(model)
    end
    if model.invalid_model
        return
    end
    copy_parameters(model)
    inner = model.inner::IpoptProblem
    # The default print level is `5`
    AddIpoptIntOption(inner, "print_level", model.silent ? 0 : 5)
    # Other misc options that over-ride the ones set above.
    for (name, value) in model.options
        if value isa String
            AddIpoptStrOption(inner, name, value)
        elseif value isa Integer
            AddIpoptIntOption(inner, name, value)
        else
            @assert value isa Float64
            AddIpoptNumOption(inner, name, value)
        end
    end
    # Initialize the starting point, projecting variables from 0 onto their
    # bounds if VariablePrimalStart  is not provided.
    for i in 1:length(model.variable_primal_start)
        inner.x[i] = if model.variable_primal_start[i] !== nothing
            model.variable_primal_start[i]
        else
            clamp(0.0, model.variables.lower[i], model.variables.upper[i])
        end
    end
    for (i, start) in enumerate(model.qp_data.mult_g)
        inner.mult_g[i] = _dual_start(model, start, -1)
    end
    offset = length(model.qp_data.mult_g)
    if model.nlp_dual_start === nothing
        inner.mult_g[(offset+1):end] .= 0.0
    else
        for (i, start) in enumerate(model.nlp_dual_start::Vector{Float64})
            inner.mult_g[offset+i] = _dual_start(model, start, -1)
        end
    end
    for i in 1:inner.n
        inner.mult_x_L[i] = _dual_start(model, model.mult_x_L[i])
        inner.mult_x_U[i] = _dual_start(model, model.mult_x_U[i], -1)
    end
    model.barrier_iterations = 0
    function _moi_callback(args...)
        # iter_count is args[2]
        model.barrier_iterations = args[2]
        if model.callback !== nothing
            return model.callback(args...)
        end
        return true
    end
    SetIntermediateCallback(inner, _moi_callback)
    IpoptSolve(inner)
    model.solve_time = time() - start_time
    return
end

# From Ipopt/src/Interfaces/IpReturnCodes_inc.h
const _STATUS_CODES = Dict(
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
    -5 => :Maximum_WallTime_Exceeded,
    -10 => :Not_Enough_Degrees_Of_Freedom,
    -11 => :Invalid_Problem_Definition,
    -12 => :Invalid_Option,
    -13 => :Invalid_Number_Detected,
    -100 => :Unrecoverable_Exception,
    -101 => :NonIpopt_Exception_Thrown,
    -102 => :Insufficient_Memory,
    -199 => :Internal_Error,
)

### MOI.ResultCount

# Ipopt always has an iterate available.
function MOI.get(model::Optimizer, ::MOI.ResultCount)
    return (model.inner !== nothing) ? 1 : 0
end

### MOI.TerminationStatus

function MOI.get(model::Optimizer, ::MOI.TerminationStatus)
    if model.invalid_model
        return MOI.INVALID_MODEL
    elseif model.inner === nothing
        return MOI.OPTIMIZE_NOT_CALLED
    end
    status = _STATUS_CODES[model.inner.status]
    if status == :Solve_Succeeded || status == :Feasible_Point_Found
        return MOI.LOCALLY_SOLVED
    elseif status == :Infeasible_Problem_Detected
        return MOI.LOCALLY_INFEASIBLE
    elseif status == :Solved_To_Acceptable_Level
        return MOI.ALMOST_LOCALLY_SOLVED
    elseif status == :Search_Direction_Becomes_Too_Small
        return MOI.NUMERICAL_ERROR
    elseif status == :Diverging_Iterates
        return MOI.NORM_LIMIT
    elseif status == :User_Requested_Stop
        return MOI.INTERRUPTED
    elseif status == :Maximum_Iterations_Exceeded
        return MOI.ITERATION_LIMIT
    elseif status == :Maximum_CpuTime_Exceeded
        return MOI.TIME_LIMIT
    elseif status == :Maximum_WallTime_Exceeded
        return MOI.TIME_LIMIT
    elseif status == :Restoration_Failed
        return MOI.NUMERICAL_ERROR
    elseif status == :Error_In_Step_Computation
        return MOI.NUMERICAL_ERROR
    elseif status == :Invalid_Option
        return MOI.INVALID_OPTION
    elseif status == :Not_Enough_Degrees_Of_Freedom
        return MOI.INVALID_MODEL
    elseif status == :Invalid_Problem_Definition
        return MOI.INVALID_MODEL
    elseif status == :Invalid_Number_Detected
        return MOI.INVALID_MODEL
    elseif status == :Unrecoverable_Exception
        return MOI.OTHER_ERROR
    elseif status == :NonIpopt_Exception_Thrown
        return MOI.OTHER_ERROR
    else
        @assert status == :Insufficient_Memory
        return MOI.MEMORY_LIMIT
    end
end

### MOI.RawStatusString

function MOI.get(model::Optimizer, ::MOI.RawStatusString)
    if model.invalid_model
        return "The model has no variable"
    elseif model.inner === nothing
        return "Optimize not called"
    else
        return string(_STATUS_CODES[model.inner.status])
    end
end

### MOI.PrimalStatus

function MOI.get(model::Optimizer, attr::MOI.PrimalStatus)
    if !(1 <= attr.result_index <= MOI.get(model, MOI.ResultCount()))
        return MOI.NO_SOLUTION
    end
    status = _STATUS_CODES[model.inner.status]
    if status == :Solve_Succeeded
        return MOI.FEASIBLE_POINT
    elseif status == :Feasible_Point_Found
        return MOI.FEASIBLE_POINT
    elseif status == :Solved_To_Acceptable_Level
        # Solutions are only guaranteed to satisfy the "acceptable" convergence
        # tolerances.
        return MOI.NEARLY_FEASIBLE_POINT
    elseif status == :Infeasible_Problem_Detected
        return MOI.INFEASIBLE_POINT
    else
        return MOI.UNKNOWN_RESULT_STATUS
    end
end

### MOI.DualStatus

function MOI.get(model::Optimizer, attr::MOI.DualStatus)
    if !(1 <= attr.result_index <= MOI.get(model, MOI.ResultCount()))
        return MOI.NO_SOLUTION
    end
    status = _STATUS_CODES[model.inner.status]
    if status == :Solve_Succeeded
        return MOI.FEASIBLE_POINT
    elseif status == :Feasible_Point_Found
        return MOI.FEASIBLE_POINT
    elseif status == :Solved_To_Acceptable_Level
        # Solutions are only guaranteed to satisfy the "acceptable" convergence
        # tolerances.
        return MOI.NEARLY_FEASIBLE_POINT
    else
        return MOI.UNKNOWN_RESULT_STATUS
    end
end

### MOI.SolveTimeSec

MOI.get(model::Optimizer, ::MOI.SolveTimeSec) = model.solve_time

### MOI.BarrierIterations

MOI.get(model::Optimizer, ::MOI.BarrierIterations) = model.barrier_iterations

### MOI.ObjectiveValue

function MOI.get(model::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(model, attr)
    return model.inner.obj_val
end

### MOI.VariablePrimal

function MOI.get(
    model::Optimizer,
    attr::MOI.VariablePrimal,
    vi::MOI.VariableIndex,
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, vi)
    if _is_parameter(vi)
        p = model.parameters[vi]
        return model.nlp_model[p]
    end
    return model.inner.x[column(vi)]
end

### MOI.ConstraintPrimal

row(model::Optimizer, ci::MOI.ConstraintIndex{<:_FUNCTIONS}) = ci.value

function row(
    model::Optimizer,
    ci::MOI.ConstraintIndex{MOI.ScalarNonlinearFunction},
)
    return length(model.qp_data) + ci.value
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{<:_FUNCTIONS,<:_SETS},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    return model.inner.g[row(model, ci)]
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,<:_SETS},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    return model.inner.x[ci.value]
end

### MOI.ConstraintDual

_dual_multiplier(model::Optimizer) = model.sense == MOI.MIN_SENSE ? 1.0 : -1.0

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{<:_FUNCTIONS,<:_SETS},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    s = -_dual_multiplier(model)
    return s * model.inner.mult_g[row(model, ci)]
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.LessThan{Float64}},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    rc = model.inner.mult_x_L[ci.value] - model.inner.mult_x_U[ci.value]
    return min(0.0, _dual_multiplier(model) * rc)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.GreaterThan{Float64}},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    rc = model.inner.mult_x_L[ci.value] - model.inner.mult_x_U[ci.value]
    return max(0.0, _dual_multiplier(model) * rc)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.EqualTo{Float64}},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    rc = model.inner.mult_x_L[ci.value] - model.inner.mult_x_U[ci.value]
    return _dual_multiplier(model) * rc
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.Interval{Float64}},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    rc = model.inner.mult_x_L[ci.value] - model.inner.mult_x_U[ci.value]
    return _dual_multiplier(model) * rc
end

### MOI.NLPBlockDual

function MOI.get(model::Optimizer, attr::MOI.NLPBlockDual)
    MOI.check_result_index_bounds(model, attr)
    s = -_dual_multiplier(model)
    return s .* model.inner.mult_g[(length(model.qp_data)+1):end]
end

### Ipopt.CallbackFunction

"""
    CallbackFunction()

A solver-dependent callback for Ipopt's IntermediateCallback.

The callback should be a function like the following:
```julia
function my_intermediate_callback(
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
)
    # ... user code ...
    return true # or `return false` to terminate the solve.
end
```

The arguments are defined in the Ipopt documentation:
https://coin-or.github.io/Ipopt/OUTPUT.html

Note: Calling `SetIntermediateCallback` will over-write this callback! Don't
call both.
"""
struct CallbackFunction <: MOI.AbstractCallback end

function MOI.set(model::Optimizer, ::CallbackFunction, f::Function)
    model.callback = f
    return
end

function MOI.get(
    model::Optimizer,
    ::MOI.CallbackVariablePrimal,
    x::MOI.VariableIndex,
)
    return model.inner.x[column(x)]
end
