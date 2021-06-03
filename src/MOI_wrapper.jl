import MathOptInterface

const MOI = MathOptInterface

mutable struct VariableInfo
    lower_bound::Float64  # May be -Inf even if has_lower_bound == true
    has_lower_bound::Bool # Implies lower_bound == Inf
    lower_bound_dual_start::Union{Nothing,Float64}
    upper_bound::Float64  # May be Inf even if has_upper_bound == true
    has_upper_bound::Bool # Implies upper_bound == Inf
    upper_bound_dual_start::Union{Nothing,Float64}
    is_fixed::Bool        # Implies lower_bound == upper_bound and !has_lower_bound and !has_upper_bound.
    start::Union{Nothing,Float64}
end

function VariableInfo()
    return VariableInfo(
        -Inf,
        false,
        nothing,
        Inf,
        false,
        nothing,
        false,
        nothing,
    )
end

mutable struct ConstraintInfo{F,S}
    func::F
    set::S
    dual_start::Union{Nothing,Float64}
end

ConstraintInfo(func, set) = ConstraintInfo(func, set, nothing)

mutable struct Optimizer <: MOI.AbstractOptimizer
    inner::Union{IpoptProblem,Nothing}

    # Problem data.
    variable_info::Vector{VariableInfo}
    nlp_data::MOI.NLPBlockData
    sense::MOI.OptimizationSense
    objective::Union{
        MOI.SingleVariable,
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
        Nothing,
    }
    linear_le_constraints::Vector{
        ConstraintInfo{MOI.ScalarAffineFunction{Float64},MOI.LessThan{Float64}},
    }
    linear_ge_constraints::Vector{
        ConstraintInfo{
            MOI.ScalarAffineFunction{Float64},
            MOI.GreaterThan{Float64},
        },
    }
    linear_eq_constraints::Vector{
        ConstraintInfo{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}},
    }
    quadratic_le_constraints::Vector{
        ConstraintInfo{
            MOI.ScalarQuadraticFunction{Float64},
            MOI.LessThan{Float64},
        },
    }
    quadratic_ge_constraints::Vector{
        ConstraintInfo{
            MOI.ScalarQuadraticFunction{Float64},
            MOI.GreaterThan{Float64},
        },
    }
    quadratic_eq_constraints::Vector{
        ConstraintInfo{
            MOI.ScalarQuadraticFunction{Float64},
            MOI.EqualTo{Float64},
        },
    }
    nlp_dual_start::Union{Nothing,Vector{Float64}}

    # Parameters.
    silent::Bool
    options::Dict{String,Any}

    # Solution attributes.
    solve_time::Float64

    callback::Union{Nothing,Function}
end

struct EmptyNLPEvaluator <: MOI.AbstractNLPEvaluator end

MOI.features_available(::EmptyNLPEvaluator) = [:Grad, :Jac, :Hess]

MOI.initialize(::EmptyNLPEvaluator, features) = nothing

MOI.eval_objective(::EmptyNLPEvaluator, x) = NaN

function MOI.eval_constraint(::EmptyNLPEvaluator, g, x)
    @assert length(g) == 0
    return
end

function MOI.eval_objective_gradient(::EmptyNLPEvaluator, g, x)
    fill!(g, 0.0)
    return
end

MOI.jacobian_structure(::EmptyNLPEvaluator) = Tuple{Int64,Int64}[]

MOI.hessian_lagrangian_structure(::EmptyNLPEvaluator) = Tuple{Int64,Int64}[]

function MOI.eval_constraint_jacobian(::EmptyNLPEvaluator, J, x)
    @assert length(J) == 0
    return
end

function MOI.eval_hessian_lagrangian(::EmptyNLPEvaluator, H, x, σ, μ)
    @assert length(H) == 0
    return
end

empty_nlp_data() = MOI.NLPBlockData([], EmptyNLPEvaluator(), false)

function Optimizer(; kwargs...)
    if length(kwargs) > 0
        @warn("""Passing optimizer attributes as keyword arguments to
        `Ipopt.Optimizer` is deprecated. Use
            MOI.set(model, MOI.RawOptimizerAttribute("key"), value)
        or
            JuMP.set_optimizer_attribute(model, "key", value)
        instead.""")
    end
    return Optimizer(
        nothing,
        [],
        empty_nlp_data(),
        MOI.FEASIBILITY_SENSE,
        nothing,
        [],
        [],
        [],
        [],
        [],
        [],
        nothing,
        false,
        Dict{String,Any}(
            # Remove when `kwargs...` support is dropped.
            string(key) => value for (key, value) in kwargs
        ),
        NaN,
        nothing,
    )
end

MOI.supports(::Optimizer, ::MOI.NLPBlock) = true

function MOI.supports(::Optimizer, ::MOI.ObjectiveFunction{MOI.SingleVariable})
    return true
end

function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}},
)
    return true
end

function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}},
)
    return true
end

MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true

MOI.supports(::Optimizer, ::MOI.Silent) = true

MOI.supports(::Optimizer, ::MOI.RawOptimizerAttribute) = true

function MOI.supports(
    ::Optimizer,
    ::MOI.VariablePrimalStart,
    ::Type{MOI.VariableIndex},
)
    return true
end

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.SingleVariable},
    ::Type{MOI.LessThan{Float64}},
)
    return true
end
function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.SingleVariable},
    ::Type{MOI.GreaterThan{Float64}},
)
    return true
end
function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.SingleVariable},
    ::Type{MOI.EqualTo{Float64}},
)
    return true
end
function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.ScalarAffineFunction{Float64}},
    ::Type{MOI.LessThan{Float64}},
)
    return true
end
function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.ScalarAffineFunction{Float64}},
    ::Type{MOI.GreaterThan{Float64}},
)
    return true
end
function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.ScalarAffineFunction{Float64}},
    ::Type{MOI.EqualTo{Float64}},
)
    return true
end
function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.ScalarQuadraticFunction{Float64}},
    ::Type{MOI.LessThan{Float64}},
)
    return true
end
function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.ScalarQuadraticFunction{Float64}},
    ::Type{MOI.GreaterThan{Float64}},
)
    return true
end
function MOI.supports_constraint(
    ::Optimizer,
    ::Type{MOI.ScalarQuadraticFunction{Float64}},
    ::Type{MOI.EqualTo{Float64}},
)
    return true
end

function MOI.Utilities.supports_default_copy_to(::Optimizer, copy_names::Bool)
    return !copy_names
end

function MOI.copy_to(model::Optimizer, src::MOI.ModelLike; copy_names = false)
    return MOI.Utilities.default_copy_to(model, src, copy_names)
end

MOI.get(::Optimizer, ::MOI.SolverName) = "Ipopt"

MOI.get(model::Optimizer, ::MOI.ObjectiveFunctionType) = typeof(model.objective)

MOI.get(model::Optimizer, ::MOI.NumberOfVariables) = length(model.variable_info)

function MOI.get(
    model::Optimizer,
    ::MOI.NumberOfConstraints{
        MOI.ScalarAffineFunction{Float64},
        MOI.LessThan{Float64},
    },
)
    return length(model.linear_le_constraints)
end
function MOI.get(
    model::Optimizer,
    ::MOI.NumberOfConstraints{
        MOI.ScalarAffineFunction{Float64},
        MOI.EqualTo{Float64},
    },
)
    return length(model.linear_eq_constraints)
end
function MOI.get(
    model::Optimizer,
    ::MOI.NumberOfConstraints{
        MOI.ScalarAffineFunction{Float64},
        MOI.GreaterThan{Float64},
    },
)
    return length(model.linear_ge_constraints)
end
function MOI.get(
    model::Optimizer,
    ::MOI.NumberOfConstraints{MOI.SingleVariable,MOI.LessThan{Float64}},
)
    return count(e -> e.has_upper_bound, model.variable_info)
end
function MOI.get(
    model::Optimizer,
    ::MOI.NumberOfConstraints{MOI.SingleVariable,MOI.EqualTo{Float64}},
)
    return count(e -> e.is_fixed, model.variable_info)
end
function MOI.get(
    model::Optimizer,
    ::MOI.NumberOfConstraints{MOI.SingleVariable,MOI.GreaterThan{Float64}},
)
    return count(e -> e.has_lower_bound, model.variable_info)
end

function MOI.get(model::Optimizer, ::MOI.ListOfVariableIndices)
    return [MOI.VariableIndex(i) for i in 1:length(model.variable_info)]
end

function MOI.get(model::Optimizer, ::MOI.ListOfConstraintTypesPresent)
    constraints = Set{Tuple{DataType,DataType}}()
    for info in model.variable_info
        if info.has_lower_bound
            push!(constraints, (MOI.SingleVariable, MOI.LessThan{Float64}))
        end
        if info.has_upper_bound
            push!(constraints, (MOI.SingleVariable, MOI.GreaterThan{Float64}))
        end
        if info.is_fixed
            push!(constraints, (MOI.SingleVariable, MOI.EqualTo{Float64}))
        end
    end
    # Handling model constraints separately
    if !isempty(model.linear_le_constraints)
        push!(
            constraints,
            (MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}),
        )
    end
    if !isempty(model.linear_ge_constraints)
        push!(
            constraints,
            (MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}),
        )
    end
    if !isempty(model.linear_eq_constraints)
        push!(
            constraints,
            (MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}),
        )
    end
    if !isempty(model.quadratic_le_constraints)
        push!(
            constraints,
            (MOI.ScalarQuadraticFunction{Float64}, MOI.LessThan{Float64}),
        )
    end
    if !isempty(model.quadratic_ge_constraints)
        push!(
            constraints,
            (MOI.ScalarQuadraticFunction{Float64}, MOI.GreaterThan{Float64}),
        )
    end
    if !isempty(model.quadratic_eq_constraints)
        push!(
            constraints,
            (MOI.ScalarQuadraticFunction{Float64}, MOI.EqualTo{Float64}),
        )
    end
    return collect(constraints)
end

function MOI.get(
    model::Optimizer,
    ::MOI.ListOfConstraintIndices{
        MOI.ScalarAffineFunction{Float64},
        MOI.LessThan{Float64},
    },
)
    return MOI.ConstraintIndex{
        MOI.ScalarAffineFunction{Float64},
        MOI.LessThan{Float64},
    }.(eachindex(model.linear_le_constraints))
end

function MOI.get(
    model::Optimizer,
    ::MOI.ListOfConstraintIndices{
        MOI.ScalarAffineFunction{Float64},
        MOI.EqualTo{Float64},
    },
)
    return MOI.ConstraintIndex{
        MOI.ScalarAffineFunction{Float64},
        MOI.EqualTo{Float64},
    }.(eachindex(model.linear_eq_constraints))
end

function MOI.get(
    model::Optimizer,
    ::MOI.ListOfConstraintIndices{
        MOI.ScalarAffineFunction{Float64},
        MOI.GreaterThan{Float64},
    },
)
    return MOI.ConstraintIndex{
        MOI.ScalarAffineFunction{Float64},
        MOI.GreaterThan{Float64},
    }.(eachindex(model.linear_ge_constraints))
end

function MOI.get(
    model::Optimizer,
    ::MOI.ListOfConstraintIndices{MOI.SingleVariable,MOI.LessThan{Float64}},
)
    dict =
        Dict(model.variable_info[i] => i for i in 1:length(model.variable_info))
    filter!(info -> info.first.has_upper_bound, dict)
    return MOI.ConstraintIndex{
        MOI.SingleVariable,
        MOI.LessThan{Float64},
    }.(values(dict))
end

function MOI.get(
    model::Optimizer,
    ::MOI.ListOfConstraintIndices{MOI.SingleVariable,MOI.EqualTo{Float64}},
)
    dict =
        Dict(model.variable_info[i] => i for i in 1:length(model.variable_info))
    filter!(info -> info.first.is_fixed, dict)
    return MOI.ConstraintIndex{
        MOI.SingleVariable,
        MOI.EqualTo{Float64},
    }.(values(dict))
end

function MOI.get(
    model::Optimizer,
    ::MOI.ListOfConstraintIndices{MOI.SingleVariable,MOI.GreaterThan{Float64}},
)
    dict =
        Dict(model.variable_info[i] => i for i in 1:length(model.variable_info))
    filter!(info -> info.first.has_lower_bound, dict)
    return MOI.ConstraintIndex{
        MOI.SingleVariable,
        MOI.GreaterThan{Float64},
    }.(values(dict))
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintFunction,
    c::MOI.ConstraintIndex{
        MOI.ScalarAffineFunction{Float64},
        MOI.LessThan{Float64},
    },
)
    return model.linear_le_constraints[c.value].func
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintFunction,
    c::MOI.ConstraintIndex{
        MOI.ScalarAffineFunction{Float64},
        MOI.EqualTo{Float64},
    },
)
    return model.linear_eq_constraints[c.value].func
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintFunction,
    c::MOI.ConstraintIndex{
        MOI.ScalarAffineFunction{Float64},
        MOI.GreaterThan{Float64},
    },
)
    return model.linear_ge_constraints[c.value].func
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintFunction,
    c::MOI.ConstraintIndex{MOI.SingleVariable,MOI.LessThan{Float64}},
)
    return MOI.SingleVariable(MOI.VariableIndex(c.value))
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintFunction,
    c::MOI.ConstraintIndex{MOI.SingleVariable,MOI.EqualTo{Float64}},
)
    return MOI.SingleVariable(MOI.VariableIndex(c.value))
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintFunction,
    c::MOI.ConstraintIndex{MOI.SingleVariable,MOI.GreaterThan{Float64}},
)
    return MOI.SingleVariable(MOI.VariableIndex(c.value))
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{
        MOI.ScalarAffineFunction{Float64},
        MOI.LessThan{Float64},
    },
)
    return model.linear_le_constraints[c.value].set
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{
        MOI.ScalarAffineFunction{Float64},
        MOI.EqualTo{Float64},
    },
)
    return model.linear_eq_constraints[c.value].set
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{
        MOI.ScalarAffineFunction{Float64},
        MOI.GreaterThan{Float64},
    },
)
    return model.linear_ge_constraints[c.value].set
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{MOI.SingleVariable,MOI.LessThan{Float64}},
)
    return MOI.LessThan{Float64}(model.variable_info[c.value].upper_bound)
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{MOI.SingleVariable,MOI.EqualTo{Float64}},
)
    return MOI.EqualTo{Float64}(model.variable_info[c.value].lower_bound)
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintSet,
    c::MOI.ConstraintIndex{MOI.SingleVariable,MOI.GreaterThan{Float64}},
)
    return MOI.GreaterThan{Float64}(model.variable_info[c.value].lower_bound)
end

function MOI.get(model::Optimizer, ::MOI.ObjectiveFunction{T})::T where {T}
    return convert(T, model.objective)
end

function MOI.set(
    model::Optimizer,
    ::MOI.ObjectiveSense,
    sense::MOI.OptimizationSense,
)
    model.sense = sense
    return
end

MOI.get(model::Optimizer, ::MOI.ObjectiveSense) = model.sense

function MOI.set(model::Optimizer, ::MOI.Silent, value)
    model.silent = value
    return
end

MOI.get(model::Optimizer, ::MOI.Silent) = model.silent

const TIME_LIMIT = "max_cpu_time"

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true

function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, value::Real)
    MOI.set(model, MOI.RawOptimizerAttribute(TIME_LIMIT), Float64(value))
    return
end

function MOI.set(model::Optimizer, attr::MOI.TimeLimitSec, ::Nothing)
    return delete!(model.options, TIME_LIMIT)
end

function MOI.get(model::Optimizer, ::MOI.TimeLimitSec)
    return get(model.options, TIME_LIMIT, nothing)
end

function MOI.set(model::Optimizer, p::MOI.RawOptimizerAttribute, value)
    model.options[p.name] = value
    return
end

function MOI.get(model::Optimizer, p::MOI.RawOptimizerAttribute)
    if !haskey(model.options, p.name)
        error("RawParameter with name $(p.name) is not set.")
    end
    return model.options[p.name]
end

MOI.get(model::Optimizer, ::MOI.SolveTimeSec) = model.solve_time

function MOI.empty!(model::Optimizer)
    model.inner = nothing
    empty!(model.variable_info)
    model.nlp_data = empty_nlp_data()
    model.sense = MOI.FEASIBILITY_SENSE
    model.objective = nothing
    empty!(model.linear_le_constraints)
    empty!(model.linear_ge_constraints)
    empty!(model.linear_eq_constraints)
    empty!(model.quadratic_le_constraints)
    empty!(model.quadratic_ge_constraints)
    empty!(model.quadratic_eq_constraints)
    return model.nlp_dual_start = nothing
end

function MOI.is_empty(model::Optimizer)
    return isempty(model.variable_info) &&
           model.nlp_data.evaluator isa EmptyNLPEvaluator &&
           model.sense == MOI.FEASIBILITY_SENSE &&
           isempty(model.linear_le_constraints) &&
           isempty(model.linear_ge_constraints) &&
           isempty(model.linear_eq_constraints) &&
           isempty(model.quadratic_le_constraints) &&
           isempty(model.quadratic_ge_constraints) &&
           isempty(model.quadratic_eq_constraints)
end

function MOI.add_variable(model::Optimizer)
    push!(model.variable_info, VariableInfo())
    return MOI.VariableIndex(length(model.variable_info))
end

function MOI.add_variables(model::Optimizer, n::Int)
    return [MOI.add_variable(model) for i in 1:n]
end

function MOI.is_valid(model::Optimizer, vi::MOI.VariableIndex)
    return column(vi) in eachindex(model.variable_info)
end

function MOI.is_valid(
    model::Optimizer,
    ci::MOI.ConstraintIndex{MOI.SingleVariable,MOI.LessThan{Float64}},
)
    vi = MOI.VariableIndex(ci.value)
    return MOI.is_valid(model, vi) && has_upper_bound(model, vi)
end

function MOI.is_valid(
    model::Optimizer,
    ci::MOI.ConstraintIndex{MOI.SingleVariable,MOI.GreaterThan{Float64}},
)
    vi = MOI.VariableIndex(ci.value)
    return MOI.is_valid(model, vi) && has_lower_bound(model, vi)
end

function MOI.is_valid(
    model::Optimizer,
    ci::MOI.ConstraintIndex{MOI.SingleVariable,MOI.EqualTo{Float64}},
)
    vi = MOI.VariableIndex(ci.value)
    return MOI.is_valid(model, vi) && is_fixed(model, vi)
end

function check_inbounds(model::Optimizer, var::MOI.SingleVariable)
    return MOI.throw_if_not_valid(model, var.variable)
end

function check_inbounds(model::Optimizer, aff::MOI.ScalarAffineFunction)
    for term in aff.terms
        MOI.throw_if_not_valid(model, term.variable)
    end
end

function check_inbounds(model::Optimizer, quad::MOI.ScalarQuadraticFunction)
    for term in quad.affine_terms
        MOI.throw_if_not_valid(model, term.variable)
    end
    for term in quad.quadratic_terms
        MOI.throw_if_not_valid(model, term.variable_1)
        MOI.throw_if_not_valid(model, term.variable_2)
    end
end

function has_upper_bound(model::Optimizer, vi::MOI.VariableIndex)
    return model.variable_info[column(vi)].has_upper_bound
end

function has_lower_bound(model::Optimizer, vi::MOI.VariableIndex)
    return model.variable_info[column(vi)].has_lower_bound
end

function is_fixed(model::Optimizer, vi::MOI.VariableIndex)
    return model.variable_info[column(vi)].is_fixed
end

function MOI.add_constraint(
    model::Optimizer,
    v::MOI.SingleVariable,
    lt::MOI.LessThan{Float64},
)
    vi = v.variable
    MOI.throw_if_not_valid(model, vi)
    if isnan(lt.upper)
        error("Invalid upper bound value $(lt.upper).")
    end
    if has_upper_bound(model, vi)
        throw(MOI.UpperBoundAlreadySet{typeof(lt),typeof(lt)}(vi))
    end
    if is_fixed(model, vi)
        throw(MOI.UpperBoundAlreadySet{MOI.EqualTo{Float64},typeof(lt)}(vi))
    end
    col = column(vi)
    model.variable_info[col].upper_bound = lt.upper
    model.variable_info[col].has_upper_bound = true
    return MOI.ConstraintIndex{MOI.SingleVariable,MOI.LessThan{Float64}}(col)
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintSet,
    ci::MOI.ConstraintIndex{MOI.SingleVariable,MOI.LessThan{Float64}},
    set::MOI.LessThan{Float64},
)
    MOI.throw_if_not_valid(model, ci)
    model.variable_info[ci.value].upper_bound = set.upper
    return
end

function MOI.delete(
    model::Optimizer,
    ci::MOI.ConstraintIndex{MOI.SingleVariable,MOI.LessThan{Float64}},
)
    MOI.throw_if_not_valid(model, ci)
    model.variable_info[ci.value].upper_bound = Inf
    model.variable_info[ci.value].has_upper_bound = false
    return
end

function MOI.add_constraint(
    model::Optimizer,
    v::MOI.SingleVariable,
    gt::MOI.GreaterThan{Float64},
)
    vi = v.variable
    MOI.throw_if_not_valid(model, vi)
    if isnan(gt.lower)
        error("Invalid lower bound value $(gt.lower).")
    end
    if has_lower_bound(model, vi)
        throw(MOI.LowerBoundAlreadySet{typeof(gt),typeof(gt)}(vi))
    end
    if is_fixed(model, vi)
        throw(MOI.LowerBoundAlreadySet{MOI.EqualTo{Float64},typeof(gt)}(vi))
    end
    col = column(vi)
    model.variable_info[col].lower_bound = gt.lower
    model.variable_info[col].has_lower_bound = true
    return MOI.ConstraintIndex{MOI.SingleVariable,MOI.GreaterThan{Float64}}(col)
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintSet,
    ci::MOI.ConstraintIndex{MOI.SingleVariable,MOI.GreaterThan{Float64}},
    set::MOI.GreaterThan{Float64},
)
    MOI.throw_if_not_valid(model, ci)
    model.variable_info[ci.value].lower_bound = set.lower
    return
end

function MOI.delete(
    model::Optimizer,
    ci::MOI.ConstraintIndex{MOI.SingleVariable,MOI.GreaterThan{Float64}},
)
    MOI.throw_if_not_valid(model, ci)
    model.variable_info[ci.value].lower_bound = -Inf
    model.variable_info[ci.value].has_lower_bound = false
    return
end

function MOI.add_constraint(
    model::Optimizer,
    v::MOI.SingleVariable,
    eq::MOI.EqualTo{Float64},
)
    vi = v.variable
    MOI.throw_if_not_valid(model, vi)
    if isnan(eq.value)
        error("Invalid fixed value $(eq.value).")
    end
    if has_lower_bound(model, vi)
        throw(MOI.LowerBoundAlreadySet{MOI.GreaterThan{Float64},typeof(eq)}(vi))
    end
    if has_upper_bound(model, vi)
        throw(MOI.UpperBoundAlreadySet{MOI.LessThan{Float64},typeof(eq)}(vi))
    end
    if is_fixed(model, vi)
        throw(MOI.LowerBoundAlreadySet{typeof(eq),typeof(eq)}(vi))
    end
    col = column(vi)
    model.variable_info[col].lower_bound = eq.value
    model.variable_info[col].upper_bound = eq.value
    model.variable_info[col].is_fixed = true
    return MOI.ConstraintIndex{MOI.SingleVariable,MOI.EqualTo{Float64}}(col)
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintSet,
    ci::MOI.ConstraintIndex{MOI.SingleVariable,MOI.EqualTo{Float64}},
    set::MOI.EqualTo{Float64},
)
    MOI.throw_if_not_valid(model, ci)
    model.variable_info[ci.value].lower_bound = set.value
    model.variable_info[ci.value].upper_bound = set.value
    return
end

function MOI.delete(
    model::Optimizer,
    ci::MOI.ConstraintIndex{MOI.SingleVariable,MOI.EqualTo{Float64}},
)
    MOI.throw_if_not_valid(model, ci)
    model.variable_info[ci.value].lower_bound = -Inf
    model.variable_info[ci.value].upper_bound = Inf
    model.variable_info[ci.value].is_fixed = false
    return
end

macro define_add_constraint(function_type, set_type, prefix)
    array_name = Symbol(string(prefix) * "_constraints")
    return quote
        function MOI.add_constraint(
            model::Optimizer,
            func::$function_type,
            set::$set_type,
        )
            check_inbounds(model, func)
            push!(model.$(array_name), ConstraintInfo(func, set))
            return MOI.ConstraintIndex{$function_type,$set_type}(
                length(model.$(array_name)),
            )
        end
    end
end

@define_add_constraint(
    MOI.ScalarAffineFunction{Float64},
    MOI.LessThan{Float64},
    linear_le,
)

@define_add_constraint(
    MOI.ScalarAffineFunction{Float64},
    MOI.GreaterThan{Float64},
    linear_ge,
)

@define_add_constraint(
    MOI.ScalarAffineFunction{Float64},
    MOI.EqualTo{Float64},
    linear_eq,
)

@define_add_constraint(
    MOI.ScalarQuadraticFunction{Float64},
    MOI.LessThan{Float64},
    quadratic_le,
)

@define_add_constraint(
    MOI.ScalarQuadraticFunction{Float64},
    MOI.GreaterThan{Float64},
    quadratic_ge,
)

@define_add_constraint(
    MOI.ScalarQuadraticFunction{Float64},
    MOI.EqualTo{Float64},
    quadratic_eq,
)

function MOI.set(
    model::Optimizer,
    ::MOI.VariablePrimalStart,
    vi::MOI.VariableIndex,
    value::Union{Real,Nothing},
)
    MOI.throw_if_not_valid(model, vi)
    model.variable_info[column(vi)].start = value
    return
end

_dual_start(::Optimizer, ::Nothing, ::Int = 1) = 0.0
function _dual_start(model::Optimizer, value::Real, scale::Int = 1)
    return _dual_multiplier(model) * value * scale
end

function MOI.supports(
    ::Optimizer,
    ::MOI.ConstraintDualStart,
    ::Type{
        MOI.ConstraintIndex{
            MOI.SingleVariable,
            <:Union{MOI.GreaterThan,MOI.LessThan,MOI.EqualTo},
        },
    },
)
    return true
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{MOI.SingleVariable,MOI.GreaterThan{Float64}},
    value::Union{Real,Nothing},
)
    MOI.throw_if_not_valid(model, ci)
    model.variable_info[ci.value].lower_bound_dual_start = value
    return
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{MOI.SingleVariable,MOI.GreaterThan{Float64}},
)
    MOI.throw_if_not_valid(model, ci)
    return model.variable_info[ci.value].lower_bound_dual_start
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{MOI.SingleVariable,MOI.LessThan{Float64}},
    value::Union{Real,Nothing},
)
    MOI.throw_if_not_valid(model, ci)
    model.variable_info[ci.value].upper_bound_dual_start = value
    return
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{MOI.SingleVariable,MOI.LessThan{Float64}},
)
    MOI.throw_if_not_valid(model, ci)
    return model.variable_info[ci.value].upper_bound_dual_start
end

function MOI.set(
    model::Optimizer,
    ::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{MOI.SingleVariable,MOI.EqualTo{Float64}},
    value::Union{Real,Nothing},
)
    MOI.throw_if_not_valid(model, ci)
    if value === nothing
        model.variable_info[ci.value].upper_bound_dual_start = nothing
        model.variable_info[ci.value].lower_bound_dual_start = nothing
    elseif value >= 0.0
        model.variable_info[ci.value].upper_bound_dual_start = 0.0
        model.variable_info[ci.value].lower_bound_dual_start = value
    else
        model.variable_info[ci.value].upper_bound_dual_start = value
        model.variable_info[ci.value].lower_bound_dual_start = 0.0
    end
    return
end

function MOI.get(
    model::Optimizer,
    ::MOI.ConstraintDualStart,
    ci::MOI.ConstraintIndex{MOI.SingleVariable,MOI.EqualTo{Float64}},
)
    MOI.throw_if_not_valid(model, ci)
    upper = model.variable_info[ci.value].upper_bound_dual_start
    lower = model.variable_info[ci.value].lower_bound_dual_start
    return (upper === lower === nothing) ? nothing : lower + upper
end

macro define_constraint_dual_start(function_type, set_type, prefix)
    array_name = Symbol("$(prefix)_constraints")
    quote
        function MOI.supports(
            ::Optimizer,
            ::MOI.ConstraintDualStart,
            ::Type{MOI.ConstraintIndex{$function_type,$set_type}},
        )
            return true
        end

        function MOI.set(
            model::Optimizer,
            ::MOI.ConstraintDualStart,
            ci::MOI.ConstraintIndex{$function_type,$set_type},
            value::Union{Real,Nothing},
        )
            if !(1 <= ci.value <= length(model.$(array_name)))
                throw(MOI.InvalidIndex(ci))
            end
            model.$array_name[ci.value].dual_start = value
            return
        end

        function MOI.get(
            model::Optimizer,
            ::MOI.ConstraintDualStart,
            ci::MOI.ConstraintIndex{$function_type,$set_type},
        )
            if !(1 <= ci.value <= length(model.$(array_name)))
                throw(MOI.InvalidIndex(ci))
            end
            return model.$array_name[ci.value].dual_start
        end
    end
end

@define_constraint_dual_start(
    MOI.ScalarAffineFunction{Float64},
    MOI.LessThan{Float64},
    linear_le,
)

@define_constraint_dual_start(
    MOI.ScalarAffineFunction{Float64},
    MOI.GreaterThan{Float64},
    linear_ge,
)

@define_constraint_dual_start(
    MOI.ScalarAffineFunction{Float64},
    MOI.EqualTo{Float64},
    linear_eq,
)

@define_constraint_dual_start(
    MOI.ScalarQuadraticFunction{Float64},
    MOI.LessThan{Float64},
    quadratic_le,
)

@define_constraint_dual_start(
    MOI.ScalarQuadraticFunction{Float64},
    MOI.GreaterThan{Float64},
    quadratic_ge,
)

@define_constraint_dual_start(
    MOI.ScalarQuadraticFunction{Float64},
    MOI.EqualTo{Float64},
    quadratic_eq,
)

function MOI.supports(::Optimizer, ::MOI.NLPBlockDualStart)
    return true
end

function MOI.set(
    model::Optimizer,
    ::MOI.NLPBlockDualStart,
    values::Union{Nothing,Vector},
)
    model.nlp_dual_start = values
    return
end

MOI.get(model::Optimizer, ::MOI.NLPBlockDualStart) = model.nlp_dual_start

function MOI.set(model::Optimizer, ::MOI.NLPBlock, nlp_data::MOI.NLPBlockData)
    model.nlp_data = nlp_data
    return
end

function MOI.set(
    model::Optimizer,
    ::MOI.ObjectiveFunction,
    func::Union{
        MOI.SingleVariable,
        MOI.ScalarAffineFunction,
        MOI.ScalarQuadraticFunction,
    },
)
    check_inbounds(model, func)
    model.objective = func
    return
end

# In setting up the data for Ipopt, we order the constraints as follows:
# - linear_le_constraints
# - linear_ge_constraints
# - linear_eq_constraints
# - quadratic_le_constraints
# - quadratic_ge_constraints
# - quadratic_eq_constraints
# - nonlinear constraints from nlp_data

linear_le_offset(model::Optimizer) = 0
linear_ge_offset(model::Optimizer) = length(model.linear_le_constraints)
function linear_eq_offset(model::Optimizer)
    return linear_ge_offset(model) + length(model.linear_ge_constraints)
end
function quadratic_le_offset(model::Optimizer)
    return linear_eq_offset(model) + length(model.linear_eq_constraints)
end
function quadratic_ge_offset(model::Optimizer)
    return quadratic_le_offset(model) + length(model.quadratic_le_constraints)
end
function quadratic_eq_offset(model::Optimizer)
    return quadratic_ge_offset(model) + length(model.quadratic_ge_constraints)
end
function nlp_constraint_offset(model::Optimizer)
    return quadratic_eq_offset(model) + length(model.quadratic_eq_constraints)
end

# Convenience functions used only in optimize!

function append_to_jacobian_sparsity!(
    jacobian_sparsity,
    aff::MOI.ScalarAffineFunction,
    row,
)
    for term in aff.terms
        push!(jacobian_sparsity, (row, term.variable.value))
    end
end

function append_to_jacobian_sparsity!(
    jacobian_sparsity,
    quad::MOI.ScalarQuadraticFunction,
    row,
)
    for term in quad.affine_terms
        push!(jacobian_sparsity, (row, term.variable.value))
    end
    for term in quad.quadratic_terms
        row_idx = term.variable_1
        col_idx = term.variable_2
        if row_idx == col_idx
            push!(jacobian_sparsity, (row, row_idx.value))
        else
            push!(jacobian_sparsity, (row, row_idx.value))
            push!(jacobian_sparsity, (row, col_idx.value))
        end
    end
end

# Refers to local variables in jacobian_structure() below.
macro append_to_jacobian_sparsity(array_name)
    escrow = esc(:row)
    quote
        for info in $(esc(array_name))
            append_to_jacobian_sparsity!(
                $(esc(:jacobian_sparsity)),
                info.func,
                $escrow,
            )
            $escrow += 1
        end
    end
end

function jacobian_structure(model::Optimizer)
    num_nlp_constraints = length(model.nlp_data.constraint_bounds)
    if num_nlp_constraints > 0
        nlp_jacobian_sparsity = MOI.jacobian_structure(model.nlp_data.evaluator)
    else
        nlp_jacobian_sparsity = []
    end

    jacobian_sparsity = Tuple{Int64,Int64}[]
    row = 1
    @append_to_jacobian_sparsity model.linear_le_constraints
    @append_to_jacobian_sparsity model.linear_ge_constraints
    @append_to_jacobian_sparsity model.linear_eq_constraints
    @append_to_jacobian_sparsity model.quadratic_le_constraints
    @append_to_jacobian_sparsity model.quadratic_ge_constraints
    @append_to_jacobian_sparsity model.quadratic_eq_constraints
    for (nlp_row, column) in nlp_jacobian_sparsity
        push!(jacobian_sparsity, (nlp_row + row - 1, column))
    end
    return jacobian_sparsity
end

function append_to_hessian_sparsity!(
    ::Any,
    ::Union{MOI.SingleVariable,MOI.ScalarAffineFunction},
)
    return nothing
end

function append_to_hessian_sparsity!(
    hessian_sparsity,
    quad::MOI.ScalarQuadraticFunction,
)
    for term in quad.quadratic_terms
        push!(hessian_sparsity, (term.variable_1.value, term.variable_2.value))
    end
end

function hessian_lagrangian_structure(model::Optimizer)
    hessian_sparsity = Tuple{Int64,Int64}[]
    if !model.nlp_data.has_objective && model.objective !== nothing
        append_to_hessian_sparsity!(hessian_sparsity, model.objective)
    end
    for info in model.quadratic_le_constraints
        append_to_hessian_sparsity!(hessian_sparsity, info.func)
    end
    for info in model.quadratic_ge_constraints
        append_to_hessian_sparsity!(hessian_sparsity, info.func)
    end
    for info in model.quadratic_eq_constraints
        append_to_hessian_sparsity!(hessian_sparsity, info.func)
    end
    nlp_hessian_sparsity =
        MOI.hessian_lagrangian_structure(model.nlp_data.evaluator)
    append!(hessian_sparsity, nlp_hessian_sparsity)
    return hessian_sparsity
end

function eval_function(var::MOI.SingleVariable, x)
    return x[var.variable.value]
end

function eval_function(aff::MOI.ScalarAffineFunction, x)
    function_value = aff.constant
    for term in aff.terms
        # Note the implicit assumtion that VariableIndex values match up with
        # x indices. This is valid because in this wrapper ListOfVariableIndices
        # is always [1, ..., NumberOfVariables].
        function_value += term.coefficient * x[term.variable.value]
    end
    return function_value
end

function eval_function(quad::MOI.ScalarQuadraticFunction, x)
    function_value = quad.constant
    for term in quad.affine_terms
        function_value += term.coefficient * x[term.variable.value]
    end
    for term in quad.quadratic_terms
        row_idx = term.variable_1
        col_idx = term.variable_2
        coefficient = term.coefficient
        if row_idx == col_idx
            function_value +=
                0.5 * coefficient * x[row_idx.value] * x[col_idx.value]
        else
            function_value += coefficient * x[row_idx.value] * x[col_idx.value]
        end
    end
    return function_value
end

function eval_objective(model::Optimizer, x)
    # The order of the conditions is important. NLP objectives override regular
    # objectives.
    if model.nlp_data.has_objective
        return MOI.eval_objective(model.nlp_data.evaluator, x)
    elseif model.objective !== nothing
        return eval_function(model.objective, x)
    else
        # No objective function set. This could happen with FEASIBILITY_SENSE.
        return 0.0
    end
end

function fill_gradient!(grad, x, var::MOI.SingleVariable)
    fill!(grad, 0.0)
    return grad[var.variable.value] = 1.0
end

function fill_gradient!(grad, x, aff::MOI.ScalarAffineFunction{Float64})
    fill!(grad, 0.0)
    for term in aff.terms
        grad[term.variable.value] += term.coefficient
    end
end

function fill_gradient!(grad, x, quad::MOI.ScalarQuadraticFunction{Float64})
    fill!(grad, 0.0)
    for term in quad.affine_terms
        grad[term.variable.value] += term.coefficient
    end
    for term in quad.quadratic_terms
        row_idx = term.variable_1
        col_idx = term.variable_2
        coefficient = term.coefficient
        if row_idx == col_idx
            grad[row_idx.value] += coefficient * x[row_idx.value]
        else
            grad[row_idx.value] += coefficient * x[col_idx.value]
            grad[col_idx.value] += coefficient * x[row_idx.value]
        end
    end
end

function eval_objective_gradient(model::Optimizer, grad, x)
    if model.nlp_data.has_objective
        MOI.eval_objective_gradient(model.nlp_data.evaluator, grad, x)
    elseif model.objective !== nothing
        fill_gradient!(grad, x, model.objective)
    else
        fill!(grad, 0.0)
    end
    return
end

# Refers to local variables in eval_constraint() below.
macro eval_function(array_name)
    escrow = esc(:row)
    quote
        for info in $(esc(array_name))
            $(esc(:g))[$escrow] = eval_function(info.func, $(esc(:x)))
            $escrow += 1
        end
    end
end

function eval_constraint(model::Optimizer, g, x)
    row = 1
    @eval_function model.linear_le_constraints
    @eval_function model.linear_ge_constraints
    @eval_function model.linear_eq_constraints
    @eval_function model.quadratic_le_constraints
    @eval_function model.quadratic_ge_constraints
    @eval_function model.quadratic_eq_constraints
    nlp_g = view(g, row:length(g))
    MOI.eval_constraint(model.nlp_data.evaluator, nlp_g, x)
    return
end

function fill_constraint_jacobian!(
    values,
    start_offset,
    x,
    aff::MOI.ScalarAffineFunction,
)
    num_coefficients = length(aff.terms)
    for i in 1:num_coefficients
        values[start_offset+i] = aff.terms[i].coefficient
    end
    return num_coefficients
end

function fill_constraint_jacobian!(
    values,
    start_offset,
    x,
    quad::MOI.ScalarQuadraticFunction,
)
    num_affine_coefficients = length(quad.affine_terms)
    for i in 1:num_affine_coefficients
        values[start_offset+i] = quad.affine_terms[i].coefficient
    end
    num_quadratic_coefficients = 0
    for term in quad.quadratic_terms
        row_idx = term.variable_1
        col_idx = term.variable_2
        coefficient = term.coefficient
        offset =
            start_offset + num_affine_coefficients + num_quadratic_coefficients
        if row_idx == col_idx
            values[offset+1] = coefficient * x[col_idx.value]
            num_quadratic_coefficients += 1
        else
            # Note that the order matches the Jacobian sparsity pattern.
            values[offset+1] = coefficient * x[col_idx.value]
            values[offset+2] = coefficient * x[row_idx.value]
            num_quadratic_coefficients += 2
        end
    end
    return num_affine_coefficients + num_quadratic_coefficients
end

# Refers to local variables in eval_constraint_jacobian() below.
macro fill_constraint_jacobian(array_name)
    esc_offset = esc(:offset)
    return quote
        for info in $(esc(array_name))
            $esc_offset += fill_constraint_jacobian!(
                $(esc(:values)),
                $esc_offset,
                $(esc(:x)),
                info.func,
            )
        end
    end
end

function eval_constraint_jacobian(model::Optimizer, values, x)
    offset = 0
    @fill_constraint_jacobian model.linear_le_constraints
    @fill_constraint_jacobian model.linear_ge_constraints
    @fill_constraint_jacobian model.linear_eq_constraints
    @fill_constraint_jacobian model.quadratic_le_constraints
    @fill_constraint_jacobian model.quadratic_ge_constraints
    @fill_constraint_jacobian model.quadratic_eq_constraints

    nlp_values = view(values, 1+offset:length(values))
    MOI.eval_constraint_jacobian(model.nlp_data.evaluator, nlp_values, x)
    return
end

function fill_hessian_lagrangian!(
    ::Any,
    ::Any,
    ::Any,
    ::Union{MOI.SingleVariable,MOI.ScalarAffineFunction,Nothing},
)
    return 0
end

function fill_hessian_lagrangian!(
    values,
    start_offset,
    scale_factor,
    quad::MOI.ScalarQuadraticFunction,
)
    for i in 1:length(quad.quadratic_terms)
        values[start_offset+i] =
            scale_factor * quad.quadratic_terms[i].coefficient
    end
    return length(quad.quadratic_terms)
end

function eval_hessian_lagrangian(
    model::Optimizer,
    values,
    x,
    obj_factor,
    lambda,
)
    offset = 0
    if !model.nlp_data.has_objective
        offset +=
            fill_hessian_lagrangian!(values, 0, obj_factor, model.objective)
    end
    for (i, info) in enumerate(model.quadratic_le_constraints)
        offset += fill_hessian_lagrangian!(
            values,
            offset,
            lambda[i+quadratic_le_offset(model)],
            info.func,
        )
    end
    for (i, info) in enumerate(model.quadratic_ge_constraints)
        offset += fill_hessian_lagrangian!(
            values,
            offset,
            lambda[i+quadratic_ge_offset(model)],
            info.func,
        )
    end
    for (i, info) in enumerate(model.quadratic_eq_constraints)
        offset += fill_hessian_lagrangian!(
            values,
            offset,
            lambda[i+quadratic_eq_offset(model)],
            info.func,
        )
    end
    nlp_values = view(values, 1+offset:length(values))
    nlp_lambda = view(lambda, 1+nlp_constraint_offset(model):length(lambda))
    MOI.eval_hessian_lagrangian(
        model.nlp_data.evaluator,
        nlp_values,
        x,
        obj_factor,
        nlp_lambda,
    )
    return
end

function constraint_bounds(model::Optimizer)
    constraint_lb = Float64[]
    constraint_ub = Float64[]
    for info in model.linear_le_constraints
        push!(constraint_lb, -Inf)
        push!(constraint_ub, info.set.upper)
    end
    for info in model.linear_ge_constraints
        push!(constraint_lb, info.set.lower)
        push!(constraint_ub, Inf)
    end
    for info in model.linear_eq_constraints
        push!(constraint_lb, info.set.value)
        push!(constraint_ub, info.set.value)
    end
    for info in model.quadratic_le_constraints
        push!(constraint_lb, -Inf)
        push!(constraint_ub, info.set.upper)
    end
    for info in model.quadratic_ge_constraints
        push!(constraint_lb, info.set.lower)
        push!(constraint_ub, Inf)
    end
    for info in model.quadratic_eq_constraints
        push!(constraint_lb, info.set.value)
        push!(constraint_ub, info.set.value)
    end
    for bound in model.nlp_data.constraint_bounds
        push!(constraint_lb, bound.lower)
        push!(constraint_ub, bound.upper)
    end
    return constraint_lb, constraint_ub
end

function MOI.optimize!(model::Optimizer)
    # TODO: Reuse model.inner for incremental solves if possible.
    num_variables = length(model.variable_info)
    num_linear_le_constraints = length(model.linear_le_constraints)
    num_linear_ge_constraints = length(model.linear_ge_constraints)
    num_linear_eq_constraints = length(model.linear_eq_constraints)
    nlp_row_offset = nlp_constraint_offset(model)
    num_quadratic_constraints =
        nlp_constraint_offset(model) - quadratic_le_offset(model)
    num_nlp_constraints = length(model.nlp_data.constraint_bounds)
    num_constraints = num_nlp_constraints + nlp_row_offset

    evaluator = model.nlp_data.evaluator
    features = MOI.features_available(evaluator)
    has_hessian = (:Hess in features)
    init_feat = [:Grad]
    has_hessian && push!(init_feat, :Hess)
    if num_nlp_constraints > 0
        push!(init_feat, :Jac)
    end

    MOI.initialize(evaluator, init_feat)
    jacobian_sparsity = jacobian_structure(model)
    hessian_sparsity = has_hessian ? hessian_lagrangian_structure(model) : []

    # Objective callback
    # TODO(odow): FEASIBILITY_SENSE could produce confusing solver output if a
    # nonzero objective is set.
    function eval_f_cb(x)
        if model.sense == MOI.FEASIBILITY_SENSE
            return 0.0
        end
        return eval_objective(model, x)
    end

    # Objective gradient callback
    function eval_grad_f_cb(x, grad_f)
        if model.sense == MOI.FEASIBILITY_SENSE
            grad_f .= zero(eltype(grad_f))
        else
            eval_objective_gradient(model, grad_f, x)
        end
        return
    end

    # Constraint value callback
    eval_g_cb(x, g) = eval_constraint(model, g, x)

    # Jacobian callback
    function eval_jac_g_cb(x, mode, rows, cols, values)
        if mode == :Structure
            for i in 1:length(jacobian_sparsity)
                rows[i] = jacobian_sparsity[i][1]
                cols[i] = jacobian_sparsity[i][2]
            end
        else
            eval_constraint_jacobian(model, values, x)
        end
        return
    end

    if has_hessian
        # Hessian callback
        function eval_h_cb(x, mode, rows, cols, obj_factor, lambda, values)
            if mode == :Structure
                for i in 1:length(hessian_sparsity)
                    rows[i] = hessian_sparsity[i][1]
                    cols[i] = hessian_sparsity[i][2]
                end
            else
                eval_hessian_lagrangian(model, values, x, obj_factor, lambda)
            end
            return
        end
    else
        eval_h_cb = nothing
    end

    x_l = [v.lower_bound for v in model.variable_info]
    x_u = [v.upper_bound for v in model.variable_info]

    constraint_lb, constraint_ub = constraint_bounds(model)

    start_time = time()

    model.inner = createProblem(
        num_variables,
        x_l,
        x_u,
        num_constraints,
        constraint_lb,
        constraint_ub,
        length(jacobian_sparsity),
        length(hessian_sparsity),
        eval_f_cb,
        eval_g_cb,
        eval_grad_f_cb,
        eval_jac_g_cb,
        eval_h_cb,
    )

    if model.sense == MOI.MIN_SENSE
        addOption(model.inner, "obj_scaling_factor", 1.0)
    elseif model.sense == MOI.MAX_SENSE
        addOption(model.inner, "obj_scaling_factor", -1.0)
    end

    # Ipopt crashes by default if NaN/Inf values are returned from the
    # evaluation callbacks. This option tells Ipopt to explicitly check for them
    # and return Invalid_Number_Detected instead. This setting may result in a
    # minor performance loss and can be overwritten by specifying
    # check_derivatives_for_naninf="no".
    addOption(model.inner, "check_derivatives_for_naninf", "yes")

    if !has_hessian
        addOption(model.inner, "hessian_approximation", "limited-memory")
    end
    if num_nlp_constraints == 0 && num_quadratic_constraints == 0
        addOption(model.inner, "jac_c_constant", "yes")
        addOption(model.inner, "jac_d_constant", "yes")
        if !model.nlp_data.has_objective
            # We turn on this option if all constraints are linear and the
            # objective is linear or quadratic. From the documentation, it's
            # unclear if it may also apply if the constraints are at most
            # quadratic.
            addOption(model.inner, "hessian_constant", "yes")
        end
    end

    # If nothing is provided, the default starting value is 0.0.
    model.inner.x = zeros(num_variables)
    for (i, v) in enumerate(model.variable_info)
        if v.start !== nothing
            model.inner.x[i] = v.start
        elseif v.has_lower_bound && v.has_upper_bound
            if 0.0 <= v.lower_bound
                model.inner.x[i] = v.lower_bound
            elseif v.upper_bound <= 0.0
                model.inner.x[i] = v.upper_bound
            end
        elseif v.has_lower_bound
            model.inner.x[i] = max(0.0, v.lower_bound)
        else
            model.inner.x[i] = min(0.0, v.upper_bound)
        end
    end

    if model.nlp_dual_start === nothing
        model.nlp_dual_start = zeros(Float64, num_nlp_constraints)
    end

    mult_g_start = [
        [info.dual_start for info in model.linear_le_constraints]
        [info.dual_start for info in model.linear_ge_constraints]
        [info.dual_start for info in model.linear_eq_constraints]
        [info.dual_start for info in model.quadratic_le_constraints]
        [info.dual_start for info in model.quadratic_ge_constraints]
        [info.dual_start for info in model.quadratic_eq_constraints]
        model.nlp_dual_start
    ]

    model.inner.mult_g =
        [_dual_start(model, start, -1) for start in mult_g_start]

    model.inner.mult_x_L = zeros(length(model.variable_info))
    model.inner.mult_x_U = zeros(length(model.variable_info))
    for (i, v) in enumerate(model.variable_info)
        model.inner.mult_x_L[i] = _dual_start(model, v.lower_bound_dual_start)
        model.inner.mult_x_U[i] =
            _dual_start(model, v.upper_bound_dual_start, -1)
    end

    if model.silent
        addOption(model.inner, "print_level", 0)
    end

    for (name, value) in model.options
        addOption(model.inner, name, value)
    end

    _set_intermediate_callback(model.inner, model.callback)

    solveProblem(model.inner)

    model.solve_time = time() - start_time
    return
end

function MOI.get(model::Optimizer, ::MOI.TerminationStatus)
    if model.inner === nothing
        return MOI.OPTIMIZE_NOT_CALLED
    end
    status = ApplicationReturnStatus[model.inner.status]
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
    elseif status == :Insufficient_Memory
        return MOI.MEMORY_LIMIT
    else
        error("Unrecognized Ipopt status $status")
    end
end

function MOI.get(model::Optimizer, ::MOI.RawStatusString)
    return string(ApplicationReturnStatus[model.inner.status])
end

# Ipopt always has an iterate available.
function MOI.get(model::Optimizer, ::MOI.ResultCount)
    return (model.inner !== nothing) ? 1 : 0
end

function MOI.get(model::Optimizer, attr::MOI.PrimalStatus)
    if !(1 <= attr.result_index <= MOI.get(model, MOI.ResultCount()))
        return MOI.NO_SOLUTION
    end
    status = ApplicationReturnStatus[model.inner.status]
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

function MOI.get(model::Optimizer, attr::MOI.DualStatus)
    if !(1 <= attr.result_index <= MOI.get(model, MOI.ResultCount()))
        return MOI.NO_SOLUTION
    end
    status = ApplicationReturnStatus[model.inner.status]
    if status == :Solve_Succeeded
        return MOI.FEASIBLE_POINT
    elseif status == :Feasible_Point_Found
        return MOI.FEASIBLE_POINT
    elseif status == :Solved_To_Acceptable_Level
        # Solutions are only guaranteed to satisfy the "acceptable" convergence
        # tolerances.
        return MOI.NEARLY_FEASIBLE_POINT
    elseif status == :Infeasible_Problem_Detected
        # TODO: What is the interpretation of the dual in this case?
        return MOI.UNKNOWN_RESULT_STATUS
    else
        return MOI.UNKNOWN_RESULT_STATUS
    end
end

function MOI.get(model::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(model, attr)
    return model.inner.obj_val
end

"""
    column(x::MOI.VariableIndex)

Return the column associated with a variable.
"""
column(x::MOI.VariableIndex) = x.value

# TODO: This is a bit off, because the variable primal should be available
# only after a solve. If model.inner is initialized but we haven't solved, then
# the primal values we return do not have the intended meaning.
function MOI.get(
    model::Optimizer,
    attr::MOI.VariablePrimal,
    vi::MOI.VariableIndex,
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, vi)
    return model.inner.x[column(vi)]
end

macro define_constraint_primal(function_type, set_type, prefix)
    constraint_array = Symbol(string(prefix) * "_constraints")
    offset_function = Symbol(string(prefix) * "_offset")
    quote
        function MOI.get(
            model::Optimizer,
            attr::MOI.ConstraintPrimal,
            ci::MOI.ConstraintIndex{$function_type,$set_type},
        )
            MOI.check_result_index_bounds(model, attr)
            if !(1 <= ci.value <= length(model.$(constraint_array)))
                error("Invalid constraint index ", ci.value)
            end
            return model.inner.g[ci.value+$offset_function(model)]
        end
    end
end

@define_constraint_primal(
    MOI.ScalarAffineFunction{Float64},
    MOI.LessThan{Float64},
    linear_le,
)

@define_constraint_primal(
    MOI.ScalarAffineFunction{Float64},
    MOI.GreaterThan{Float64},
    linear_ge,
)

@define_constraint_primal(
    MOI.ScalarAffineFunction{Float64},
    MOI.EqualTo{Float64},
    linear_eq,
)

@define_constraint_primal(
    MOI.ScalarQuadraticFunction{Float64},
    MOI.LessThan{Float64},
    quadratic_le,
)

@define_constraint_primal(
    MOI.ScalarQuadraticFunction{Float64},
    MOI.GreaterThan{Float64},
    quadratic_ge,
)

@define_constraint_primal(
    MOI.ScalarQuadraticFunction{Float64},
    MOI.EqualTo{Float64},
    quadratic_eq,
)

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{
        MOI.SingleVariable,
        <:Union{
            MOI.LessThan{Float64},
            MOI.GreaterThan{Float64},
            MOI.EqualTo{Float64},
        },
    },
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    return model.inner.x[ci.value]
end

_dual_multiplier(model::Optimizer) = model.sense == MOI.MIN_SENSE ? 1.0 : -1.0

macro define_constraint_dual(function_type, set_type, prefix)
    constraint_array = Symbol("$(prefix)_constraints")
    offset_function = Symbol("$(prefix)_offset")
    quote
        function MOI.get(
            model::Optimizer,
            attr::MOI.ConstraintDual,
            ci::MOI.ConstraintIndex{$function_type,$set_type},
        )
            MOI.check_result_index_bounds(model, attr)
            if !(1 <= ci.value <= length(model.$(constraint_array)))
                error("Invalid constraint index ", ci.value)
            end
            s = -_dual_multiplier(model)
            return s * model.inner.mult_g[ci.value+$offset_function(model)]
        end
    end
end

@define_constraint_dual(
    MOI.ScalarAffineFunction{Float64},
    MOI.LessThan{Float64},
    linear_le,
)

@define_constraint_dual(
    MOI.ScalarAffineFunction{Float64},
    MOI.GreaterThan{Float64},
    linear_ge,
)

@define_constraint_dual(
    MOI.ScalarAffineFunction{Float64},
    MOI.EqualTo{Float64},
    linear_eq,
)

@define_constraint_dual(
    MOI.ScalarQuadraticFunction{Float64},
    MOI.LessThan{Float64},
    quadratic_le,
)

@define_constraint_dual(
    MOI.ScalarQuadraticFunction{Float64},
    MOI.GreaterThan{Float64},
    quadratic_ge,
)

@define_constraint_dual(
    MOI.ScalarQuadraticFunction{Float64},
    MOI.EqualTo{Float64},
    quadratic_eq,
)

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.SingleVariable,MOI.LessThan{Float64}},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    rc = model.inner.mult_x_L[ci.value] - model.inner.mult_x_U[ci.value]
    return min(0.0, _dual_multiplier(model) * rc)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.SingleVariable,MOI.GreaterThan{Float64}},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    rc = model.inner.mult_x_L[ci.value] - model.inner.mult_x_U[ci.value]
    return max(0.0, _dual_multiplier(model) * rc)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.SingleVariable,MOI.EqualTo{Float64}},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    rc = model.inner.mult_x_L[ci.value] - model.inner.mult_x_U[ci.value]
    return _dual_multiplier(model) * rc
end

function MOI.get(model::Optimizer, attr::MOI.NLPBlockDual)
    MOI.check_result_index_bounds(model, attr)
    s = -_dual_multiplier(model)
    return s .* model.inner.mult_g[(1+nlp_constraint_offset(model)):end]
end

"""
    CallbackFunction()

A solver-dependent callback for Ipopt's IntermediateCallback.

The callback should be a function like the following:
```julia
function my_intermediate_callback(
    prob::IpoptProblem,
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

`prob` is an `IpoptProblem`, the object defining the low-level interface. Use
`prob.x` to obtain the current primal iterate as a vector. Use
`column(x::MOI.VariableIndex)` to map a `MOI.VariableIndex` to the 1-based
column index.

The remainder of the arguments are defined in the Ipopt documentation:
https://coin-or.github.io/Ipopt/OUTPUT.html

Note: Calling `setIntermediateCallback` will over-write this callback! Don't
call both.
"""
struct CallbackFunction <: MOI.AbstractCallback end

function _callback_function_wrapper(
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
    s_trials::Cint,
    p_problem::Ptr{Cvoid},
)
    prob = unsafe_pointer_to_objref(p_problem)::IpoptProblem
    keep_going = prob.intermediate(
        prob,
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
        s_trials,
    )
    return keep_going ? Cint(1) : Cint(0)
end

function MOI.set(model::Optimizer, ::CallbackFunction, f::Function)
    model.callback = f
    return
end

function _set_intermediate_callback(prob::IpoptProblem, callback::Nothing)
    prob.intermediate = nothing
    return
end

function _set_intermediate_callback(prob::IpoptProblem, callback::Function)
    prob.intermediate = callback
    ipopt_callback = @cfunction(
        _callback_function_wrapper,
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
        ipopt_callback,
    )
    if ret == 0
        error("IPOPT: Something went wrong setting `CallbackFunction`.")
    end
    return
end
