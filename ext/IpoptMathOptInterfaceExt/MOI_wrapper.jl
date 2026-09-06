# Copyright (c) 2013: Iain Dunning, Miles Lubin, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

"""
    Optimizer()

Create a new Ipopt optimizer.
"""
mutable struct Optimizer <: MOI.AbstractOptimizer
    inner::Union{Nothing,Ipopt.IpoptProblem}
    name::String
    invalid_model::Bool
    silent::Bool
    options::Dict{String,Any}
    solve_time::Float64
    model::MOI.ModelLike
    variable_primal_start::Vector{Union{Nothing,Float64}}
    mult_x_L::Vector{Union{Nothing,Float64}}
    mult_x_U::Vector{Union{Nothing,Float64}}
    nlp_data::MOI.NLPBlockData
    # Whether `nlp_data` was set through the legacy `MOI.NLPBlock` API, in
    # which case it must not be rebuilt from the inner nonlinear model.
    uses_nlp_block::Bool
    nlp_dual_start::Union{Nothing,Vector{Float64}}
    # The evaluator of `model`, rebuilt in `_setup_model`.
    evaluator::Union{Nothing,MOI.AbstractNLPEvaluator}
    callback::Union{Nothing,Function}
    barrier_iterations::Int
    ad_backend::MOI.Nonlinear.AbstractAutomaticDifferentiation
    jacobian_sparsity::Vector{Tuple{Int,Int}}
    hessian_sparsity::Union{Nothing,Vector{Tuple{Int,Int}}}
    needs_new_inner::Bool
    has_only_linear_constraints::Bool

    function Optimizer()
        backend = MOI.Nonlinear.SparseReverseMode()
        return new(
            nothing,
            "",
            false,
            false,
            Dict{String,Any}(),
            NaN,
            MOI.Nonlinear.model(backend),
            Union{Nothing,Float64}[],
            Union{Nothing,Float64}[],
            Union{Nothing,Float64}[],
            MOI.NLPBlockData([], _EmptyNLPEvaluator(), false),
            false,
            nothing,
            nothing,
            nothing,
            0,
            backend,
            Tuple{Int,Int}[],
            nothing,
            true,
            false,
        )
    end
end

const _SETS = Union{
    MOI.GreaterThan{Float64},
    MOI.LessThan{Float64},
    MOI.EqualTo{Float64},
    MOI.Interval{Float64},
}

MOI.get(::Optimizer, ::MOI.SolverVersion) = string(Ipopt.GetIpoptVersion())

### _EmptyNLPEvaluator

struct _EmptyNLPEvaluator <: MOI.AbstractNLPEvaluator end

MOI.features_available(::_EmptyNLPEvaluator) = [:Grad, :Jac, :Hess]
MOI.initialize(::_EmptyNLPEvaluator, ::Any) = nothing
MOI.eval_constraint(::_EmptyNLPEvaluator, g, x) = nothing
MOI.jacobian_structure(::_EmptyNLPEvaluator) = Tuple{Int64,Int64}[]
MOI.hessian_lagrangian_structure(::_EmptyNLPEvaluator) = Tuple{Int64,Int64}[]
MOI.eval_constraint_jacobian(::_EmptyNLPEvaluator, J, x) = nothing
MOI.eval_hessian_lagrangian(::_EmptyNLPEvaluator, H, x, σ, μ) = nothing

struct _NLPBlockEvaluator <: MOI.AbstractNLPEvaluator
    data::MOI.NLPBlockData
    sense::MOI.OptimizationSense
end

MOI.features_available(d::_NLPBlockEvaluator) =
    MOI.features_available(d.data.evaluator)
MOI.initialize(d::_NLPBlockEvaluator, features) =
    MOI.initialize(d.data.evaluator, features)
MOI.Nonlinear._constraint_bounds(d::_NLPBlockEvaluator) =
    d.data.constraint_bounds
MOI.Nonlinear._has_objective(d::_NLPBlockEvaluator) = d.data.has_objective
MOI.eval_objective(d::_NLPBlockEvaluator, x) =
    MOI.Nonlinear._objective_sign(d.sense) *
    MOI.eval_objective(d.data.evaluator, x)
function MOI.eval_objective_gradient(d::_NLPBlockEvaluator, g, x)
    MOI.eval_objective_gradient(d.data.evaluator, g, x)
    g .*= MOI.Nonlinear._objective_sign(d.sense)
    return
end
MOI.eval_constraint(d::_NLPBlockEvaluator, g, x) =
    MOI.eval_constraint(d.data.evaluator, g, x)
MOI.jacobian_structure(d::_NLPBlockEvaluator) =
    MOI.jacobian_structure(d.data.evaluator)
MOI.eval_constraint_jacobian(d::_NLPBlockEvaluator, J, x) =
    MOI.eval_constraint_jacobian(d.data.evaluator, J, x)
MOI.hessian_lagrangian_structure(d::_NLPBlockEvaluator) =
    MOI.hessian_lagrangian_structure(d.data.evaluator)
function MOI.eval_hessian_lagrangian(d::_NLPBlockEvaluator, H, x, σ, μ)
    sign = MOI.Nonlinear._objective_sign(d.sense)
    return MOI.eval_hessian_lagrangian(d.data.evaluator, H, x, sign * σ, μ)
end

function MOI.empty!(model::Optimizer)
    model.inner = nothing
    # SKIP: model.name
    model.invalid_model = false
    # SKIP: model.silent
    # SKIP: model.options
    model.solve_time = 0.0
    model.model = MOI.Nonlinear.model(model.ad_backend)
    empty!(model.variable_primal_start)
    empty!(model.mult_x_L)
    empty!(model.mult_x_U)
    model.nlp_data = MOI.NLPBlockData([], _EmptyNLPEvaluator(), false)
    model.uses_nlp_block = false
    model.nlp_dual_start = nothing
    model.evaluator = nothing
    model.callback = nothing
    model.barrier_iterations = 0
    # SKIP: model.ad_backend
    empty!(model.jacobian_sparsity)
    model.hessian_sparsity = nothing
    model.needs_new_inner = true
    model.has_only_linear_constraints = false
    return
end

function MOI.is_empty(model::Optimizer)
    return MOI.get(model.model, MOI.NumberOfVariables()) == 0 &&
           isempty(model.variable_primal_start) &&
           isempty(model.mult_x_L) &&
           isempty(model.mult_x_U) &&
           model.nlp_data.evaluator isa _EmptyNLPEvaluator &&
           MOI.get(model.model, MOI.ObjectiveSense()) == MOI.FEASIBILITY_SENSE
end

MOI.supports_incremental_interface(::Optimizer) = true

function MOI.copy_to(model::Optimizer, src::MOI.ModelLike)
    return MOI.Utilities.default_copy_to(model, src)
end

MOI.get(::Optimizer, ::MOI.SolverName) = "Ipopt"

MOI.supports_add_constrained_variable(
    model::Optimizer,
    S::Type{<:MOI.AbstractScalarSet},
) =
    MOI.supports_add_constrained_variable(model.model, S)

function MOI.add_constrained_variable(model::Optimizer, set::MOI.AbstractScalarSet)
    model.inner = nothing
    x = MOI.add_variable(model)
    return x, MOI.add_constraint(model.model, x, set)
end

function MOI.add_constrained_variable(
    model::Optimizer,
    set::MOI.Parameter{Float64},
)
    if model.uses_nlp_block
        error("Cannot mix the new and legacy nonlinear APIs")
    end
    model.inner = nothing
    return MOI.add_constrained_variable(model.model, set)
end

MOI.supports_constraint(
    model::Optimizer,
    F::Type{<:MOI.AbstractFunction},
    S::Type{<:MOI.AbstractSet},
) =
    MOI.supports_constraint(model.model, F, S)

MOI.get(model::Optimizer, attr::MOI.ListOfConstraintTypesPresent) =
    MOI.get(model.model, attr)

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
        msg = "RawOptimizerAttribute with name $(p.name) is not already set."
        throw(MOI.GetAttributeNotAllowed(p, msg))
    end
    return model.options[p.name]
end

### Model data forwarding

Ipopt.column(x::MOI.VariableIndex) = x.value

function MOI.add_variable(model::Optimizer)
    push!(model.variable_primal_start, nothing)
    push!(model.mult_x_L, nothing)
    push!(model.mult_x_U, nothing)
    model.inner = nothing
    return MOI.add_variable(model.model)
end

MOI.is_valid(
    model::Optimizer,
    index::Union{MOI.VariableIndex,MOI.ConstraintIndex},
) = MOI.is_valid(model.model, index)
MOI.get(model::Optimizer, attr::MOI.ListOfVariableIndices) = MOI.get(model.model, attr)
MOI.get(model::Optimizer, attr::MOI.NumberOfVariables) = MOI.get(model.model, attr)
MOI.get(model::Optimizer, attr::Union{MOI.NumberOfConstraints,MOI.ListOfConstraintIndices}) = MOI.get(model.model, attr)
MOI.get(model::Optimizer, attr::Union{MOI.ConstraintFunction,MOI.ConstraintSet}, ci::MOI.ConstraintIndex) = MOI.get(model.model, attr, ci)

function MOI.add_constraint(model::Optimizer, f::MOI.AbstractFunction, s::MOI.AbstractSet)
    if model.uses_nlp_block && MOI.Nonlinear._is_nonlinear_input(model.model, f, s)
        error("Cannot mix the new and legacy nonlinear APIs")
    end
    model.inner = nothing
    return MOI.add_constraint(model.model, f, s)
end

function MOI.set(model::Optimizer, attr::MOI.ConstraintSet, ci::MOI.ConstraintIndex, set)
    MOI.set(model.model, attr, ci, set)
    model.needs_new_inner = true
    return
end

function MOI.delete(model::Optimizer, ci::MOI.ConstraintIndex)
    MOI.delete(model.model, ci)
    model.inner = nothing
    return
end

function MOI.supports(model::Optimizer, attr::MOI.ConstraintDualStart, CI::Type{<:MOI.ConstraintIndex})
    return MOI.supports(model.model, attr, CI)
end
MOI.get(model::Optimizer, attr::MOI.ConstraintDualStart, ci::MOI.ConstraintIndex) = MOI.get(model.model, attr, ci)
function MOI.set(model::Optimizer, attr::MOI.ConstraintDualStart, ci::MOI.ConstraintIndex, value)
    MOI.set(model.model, attr, ci, value)
    return
end

MOI.supports(model::Optimizer, attr::MOI.UserDefinedFunction) = MOI.supports(model.model, attr)
function MOI.set(model::Optimizer, attr::MOI.UserDefinedFunction, value)
    return MOI.set(model.model, attr, value)
end
MOI.get(model::Optimizer, attr::MOI.ListOfSupportedNonlinearOperators) = MOI.get(model.model, attr)

### MOI.VariablePrimalStart

function MOI.supports(
    ::Optimizer,
    ::MOI.VariablePrimalStart,
    ::Type{MOI.VariableIndex},
)
    return true
end

function MOI.get(
    model::Optimizer,
    attr::MOI.VariablePrimalStart,
    vi::MOI.VariableIndex,
)
    if MOI.Nonlinear._is_parameter(vi)
        throw(MOI.GetAttributeNotAllowed(attr, "Variable is a Parameter"))
    end
    MOI.throw_if_not_valid(model, vi)
    return model.variable_primal_start[Ipopt.column(vi)]
end

function MOI.set(
    model::Optimizer,
    attr::MOI.VariablePrimalStart,
    vi::MOI.VariableIndex,
    value::Union{Real,Nothing},
)
    if MOI.Nonlinear._is_parameter(vi)
        throw(MOI.SetAttributeNotAllowed(attr, "Variable is a Parameter"))
    end
    MOI.throw_if_not_valid(model, vi)
    model.variable_primal_start[Ipopt.column(vi)] = value
    # No need to reset model.inner, because this gets handled in optimize!.
    return
end

### MOI.ConstraintDualStart

_dual_start(::Optimizer, ::Nothing, ::Int = 1) = 0.0

function _dual_start(model::Optimizer, value::Real, scale::Int = 1)
    return value * scale
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
    ci::MOI.ConstraintIndex{MOI.VariableIndex,S},
    value::Union{Real,Nothing},
) where {S<:Union{MOI.EqualTo{Float64},MOI.Interval{Float64}}}
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
    ci::MOI.ConstraintIndex{MOI.VariableIndex,S},
) where {S<:Union{MOI.EqualTo{Float64},MOI.Interval{Float64}}}
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
    if MOI.Nonlinear._has_nonlinear_data(model.model)
        error("Cannot mix the new and legacy nonlinear APIs")
    end
    model.nlp_data = nlp_data
    model.uses_nlp_block = !(nlp_data.evaluator isa _EmptyNLPEvaluator)
    model.inner = nothing
    return
end

### Objective forwarding

MOI.supports(model::Optimizer, attr::MOI.ObjectiveSense) = MOI.supports(model.model, attr)
MOI.get(model::Optimizer, attr::MOI.ObjectiveSense) = MOI.get(model.model, attr)
function MOI.set(model::Optimizer, attr::MOI.ObjectiveSense, sense::MOI.OptimizationSense)
    MOI.set(model.model, attr, sense)
    model.needs_new_inner = true
    return
end

MOI.get(model::Optimizer, attr::MOI.ObjectiveFunctionType) = MOI.get(model.model, attr)
MOI.supports(model::Optimizer, attr::MOI.ObjectiveFunction) = MOI.supports(model.model, attr)
MOI.get(model::Optimizer, attr::MOI.ObjectiveFunction) = MOI.get(model.model, attr)
function MOI.set(model::Optimizer, attr::MOI.ObjectiveFunction, f)
    if model.uses_nlp_block &&
       MOI.Nonlinear._is_nonlinear_objective(model.model, f)
        error("Cannot mix the new and legacy nonlinear APIs")
    end
    MOI.set(model.model, attr, f)
    model.inner = nothing
    return
end

### Evaluator forwarding

MOI.eval_objective(model::Optimizer, x) = MOI.eval_objective(model.evaluator, x)
MOI.eval_objective_gradient(model::Optimizer, g, x) = MOI.eval_objective_gradient(model.evaluator, g, x)
MOI.eval_constraint(model::Optimizer, g, x) = MOI.eval_constraint(model.evaluator, g, x)
MOI.jacobian_structure(model::Optimizer) = MOI.jacobian_structure(model.evaluator)
MOI.eval_constraint_jacobian(model::Optimizer, J, x) = MOI.eval_constraint_jacobian(model.evaluator, J, x)
MOI.hessian_lagrangian_structure(model::Optimizer) = MOI.hessian_lagrangian_structure(model.evaluator)
MOI.eval_hessian_lagrangian(model::Optimizer, H, x, sigma, mu) = MOI.eval_hessian_lagrangian(model.evaluator, H, x, sigma, mu)

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
    # Setting the backend will invalidate the model if it is different. But we
    # don't requrire == for `::MOI.Nonlinear.AutomaticDifferentiationBackend` so
    # act defensive and invalidate regardless.
    model.inner = nothing
    if MOI.get(model.model, MOI.NumberOfVariables()) != 0 ||
       MOI.get(model.model, MOI.ObjectiveSense()) != MOI.FEASIBILITY_SENSE
        error(
            "The automatic-differentiation backend must be set before " *
            "adding model data.",
        )
    end
    model.model = MOI.Nonlinear.model(backend)
    model.ad_backend = backend
    return
end

### MOI.optimize!

function _eval_jac_g_cb(model, x, rows, cols, values)
    if values === nothing
        for i in 1:length(model.jacobian_sparsity)
            rows[i], cols[i] = model.jacobian_sparsity[i]
        end
    else
        MOI.eval_constraint_jacobian(model, values, x)
    end
    return
end

function _eval_h_cb(model, x, rows, cols, obj_factor, lambda, values)
    if values === nothing
        for (i, v) in enumerate(model.hessian_sparsity::Vector{Tuple{Int,Int}})
            rows[i], cols[i] = v
        end
    else
        MOI.eval_hessian_lagrangian(model, values, x, obj_factor, lambda)
    end
    return
end

function _setup_inner(model::Optimizer)::Ipopt.IpoptProblem
    if !model.needs_new_inner
        return model.inner
    end
    bounds = MOI.Nonlinear._constraint_bounds(model.evaluator)
    g_L = Float64[b.lower for b in bounds]
    g_U = Float64[b.upper for b in bounds]
    function eval_h_cb(x, rows, cols, obj_factor, lambda, values)
        return _eval_h_cb(model, x, rows, cols, obj_factor, lambda, values)
    end
    has_hessian = model.hessian_sparsity !== nothing
    x_L, x_U = MOI.Nonlinear._variable_bounds(model.model)
    model.inner = Ipopt.CreateIpoptProblem(
        length(x_L),
        x_L,
        x_U,
        length(g_L),
        g_L,
        g_U,
        length(model.jacobian_sparsity),
        has_hessian ? length(model.hessian_sparsity) : 0,
        (x) -> MOI.eval_objective(model, x),
        (x, g) -> MOI.eval_constraint(model, g, x),
        (x, grad_f) -> MOI.eval_objective_gradient(model, grad_f, x),
        (x, rows, cols, values) ->
            _eval_jac_g_cb(model, x, rows, cols, values),
        has_hessian ? eval_h_cb : nothing,
    )
    inner = model.inner::Ipopt.IpoptProblem
    # Ipopt crashes by default if NaN/Inf values are returned from the
    # evaluation callbacks. This option tells Ipopt to explicitly check for them
    # and return Invalid_Number_Detected instead. This setting may result in a
    # minor performance loss and can be overwritten by specifying
    # check_derivatives_for_naninf="no".
    Ipopt.AddIpoptStrOption(inner, "check_derivatives_for_naninf", "yes")
    if !has_hessian
        Ipopt.AddIpoptStrOption(
            inner,
            "hessian_approximation",
            "limited-memory",
        )
    end
    if model.has_only_linear_constraints
        Ipopt.AddIpoptStrOption(inner, "jac_c_constant", "yes")
        Ipopt.AddIpoptStrOption(inner, "jac_d_constant", "yes")
        if !model.nlp_data.has_objective
            Ipopt.AddIpoptStrOption(inner, "hessian_constant", "yes")
        end
    end
    function _moi_callback(args...)
        # iter_count is args[2]
        model.barrier_iterations = args[2]
        if model.callback !== nothing
            return model.callback(args...)
        end
        return true
    end
    Ipopt.SetIntermediateCallback(inner, _moi_callback)
    model.needs_new_inner = false
    return model.inner
end

function _setup_model(model::Optimizer)
    if MOI.get(model, MOI.NumberOfVariables()) == 0
        # Don't attempt to create a problem because Ipopt will error.
        model.invalid_model = true
        return
    end
    vars = MOI.get(model.model, MOI.ListOfVariableIndices())
    if model.uses_nlp_block
        if !(model.model isa MOI.Nonlinear.ModelWithQuad)
            error(
                "The legacy `MOI.NLPBlock` interface cannot be combined " *
                "with the selected automatic-differentiation backend.",
            )
        end
        oracles = model.model.inner
        inner = MOI.Nonlinear.EvaluatorWithOracles(
            oracles,
            _NLPBlockEvaluator(
                model.nlp_data,
                MOI.get(model.model, MOI.ObjectiveSense()),
            ),
            vars,
        )
        model.evaluator = MOI.Nonlinear.EvaluatorWithQuad(model.model, inner)
    else
        model.evaluator =
            MOI.Nonlinear.Evaluator(model.model, model.ad_backend, vars)
    end
    has_hessian = :Hess in MOI.features_available(model.evaluator)
    has_constraints = !isempty(MOI.Nonlinear._constraint_bounds(model.evaluator))
    init_feat = [:Grad]
    if has_hessian
        push!(init_feat, :Hess)
    end
    if has_constraints
        push!(init_feat, :Jac)
    end
    MOI.initialize(model.evaluator, init_feat)
    model.jacobian_sparsity = MOI.jacobian_structure(model)
    model.hessian_sparsity = nothing
    if has_hessian
        model.hessian_sparsity = MOI.hessian_lagrangian_structure(model)
    end
    model.has_only_linear_constraints = false
    model.needs_new_inner = true
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
    inner = _setup_inner(model)
    # The default print level is `5`
    Ipopt.AddIpoptIntOption(inner, "print_level", model.silent ? 0 : 5)
    # Other misc options that over-ride the ones set above.
    for (name, value) in model.options
        if value isa String
            Ipopt.AddIpoptStrOption(inner, name, value)
        elseif value isa Integer
            Ipopt.AddIpoptIntOption(inner, name, value)
        elseif value isa Float64
            Ipopt.AddIpoptNumOption(inner, name, value)
        else
            error(
                "Unable to add option `\"$name\"` with the value " *
                "`$value::$(typeof(value))`. The value must be a `::String`, " *
                "`::Integer`, or `::Float64`.",
            )
        end
    end
    # Initialize the starting point, projecting variables from 0 onto their
    # bounds if VariablePrimalStart is not provided.
    x_L, x_U = MOI.Nonlinear._variable_bounds(model.model)
    for i in 1:length(model.variable_primal_start)
        inner.x[i] = something(
            model.variable_primal_start[i],
            clamp(0.0, x_L[i], x_U[i]),
        )
    end
    inner.mult_g .= 0.0
    starts = MOI.Nonlinear.constraint_dual_starts(model.model)
    for (i, start) in enumerate(starts)
        if start !== nothing
            inner.mult_g[i] = _dual_start(model, start, -1)
        end
    end
    if model.uses_nlp_block && model.nlp_dual_start !== nothing
        offset = length(starts)
        for (i, start) in enumerate(model.nlp_dual_start::Vector{Float64})
            inner.mult_g[offset+i] = _dual_start(model, start, -1)
        end
    end
    for i in 1:inner.n
        inner.mult_x_L[i] = _dual_start(model, model.mult_x_L[i])
        inner.mult_x_U[i] = _dual_start(model, model.mult_x_U[i], -1)
    end
    model.barrier_iterations = 0
    Ipopt.IpoptSolve(inner)
    model.solve_time = time() - start_time
    return
end

#!format:off
const _STATUS_CODES = Dict{
    Ipopt.ApplicationReturnStatus,         Tuple{MOI.TerminationStatusCode, MOI.ResultStatusCode}
}(
    Ipopt.Solve_Succeeded                    => (MOI.LOCALLY_SOLVED,        MOI.FEASIBLE_POINT),
    Ipopt.Solved_To_Acceptable_Level         => (MOI.ALMOST_LOCALLY_SOLVED, MOI.NEARLY_FEASIBLE_POINT),
    Ipopt.Infeasible_Problem_Detected        => (MOI.LOCALLY_INFEASIBLE,    MOI.INFEASIBLE_POINT),
    Ipopt.Search_Direction_Becomes_Too_Small => (MOI.SLOW_PROGRESS,         MOI.UNKNOWN_RESULT_STATUS),
    Ipopt.Diverging_Iterates                 => (MOI.NORM_LIMIT,            MOI.UNKNOWN_RESULT_STATUS),
    Ipopt.User_Requested_Stop                => (MOI.INTERRUPTED,           MOI.UNKNOWN_RESULT_STATUS),
    Ipopt.Feasible_Point_Found               => (MOI.LOCALLY_SOLVED,        MOI.FEASIBLE_POINT),
    Ipopt.Maximum_Iterations_Exceeded        => (MOI.ITERATION_LIMIT,       MOI.UNKNOWN_RESULT_STATUS),
    Ipopt.Restoration_Failed                 => (MOI.OTHER_ERROR,           MOI.UNKNOWN_RESULT_STATUS),
    Ipopt.Error_In_Step_Computation          => (MOI.NUMERICAL_ERROR,       MOI.UNKNOWN_RESULT_STATUS),
    Ipopt.Maximum_CpuTime_Exceeded           => (MOI.TIME_LIMIT,            MOI.UNKNOWN_RESULT_STATUS),
    Ipopt.Maximum_WallTime_Exceeded          => (MOI.TIME_LIMIT,            MOI.UNKNOWN_RESULT_STATUS),
    Ipopt.Not_Enough_Degrees_Of_Freedom      => (MOI.INVALID_MODEL,         MOI.UNKNOWN_RESULT_STATUS),
    Ipopt.Invalid_Problem_Definition         => (MOI.INVALID_MODEL,         MOI.UNKNOWN_RESULT_STATUS),
    Ipopt.Invalid_Option                     => (MOI.INVALID_OPTION,        MOI.UNKNOWN_RESULT_STATUS),
    Ipopt.Invalid_Number_Detected            => (MOI.INVALID_MODEL,         MOI.UNKNOWN_RESULT_STATUS),
    Ipopt.Unrecoverable_Exception            => (MOI.OTHER_ERROR,           MOI.UNKNOWN_RESULT_STATUS),
    Ipopt.NonIpopt_Exception_Thrown          => (MOI.OTHER_ERROR,           MOI.UNKNOWN_RESULT_STATUS),
    Ipopt.Insufficient_Memory                => (MOI.MEMORY_LIMIT,          MOI.UNKNOWN_RESULT_STATUS),
    Ipopt.Internal_Error                     => (MOI.OTHER_ERROR,           MOI.UNKNOWN_RESULT_STATUS),
)
#!format:on

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
    status, _ = _STATUS_CODES[Ipopt.ApplicationReturnStatus(model.inner.status)]
    return status
end

### MOI.RawStatusString

function MOI.get(model::Optimizer, ::MOI.RawStatusString)
    if model.invalid_model
        return "The model has no variable"
    elseif model.inner === nothing
        return "Optimize not called"
    end
    return string(Ipopt.ApplicationReturnStatus(model.inner.status))
end

### MOI.PrimalStatus

function _manually_evaluated_primal_status(model::Optimizer)
    x, g = model.inner.x, model.inner.g
    x_L, x_U = MOI.Nonlinear._variable_bounds(model.model)
    bounds = MOI.Nonlinear._constraint_bounds(model.evaluator)
    g_L = Float64[b.lower for b in bounds]
    g_U = Float64[b.upper for b in bounds]
    m, n = length(g_L), length(x)
    # 1e-8 is the default tolerance
    tol = get(model.options, "tol", 1e-8)
    if all(x_L[i] - tol <= x[i] <= x_U[i] + tol for i in 1:n) &&
       all(g_L[i] - tol <= g[i] <= g_U[i] + tol for i in 1:m)
        return MOI.FEASIBLE_POINT
    end
    # 1e-6 is the default acceptable tolerance
    atol = get(model.options, "acceptable_tol", 1e-6)
    if all(x_L[i] - atol <= x[i] <= x_U[i] + atol for i in 1:n) &&
       all(g_L[i] - atol <= g[i] <= g_U[i] + atol for i in 1:m)
        return MOI.NEARLY_FEASIBLE_POINT
    end
    return MOI.INFEASIBLE_POINT
end

function MOI.get(model::Optimizer, attr::MOI.PrimalStatus)
    if !(1 <= attr.result_index <= MOI.get(model, MOI.ResultCount()))
        return MOI.NO_SOLUTION
    end
    _, status = _STATUS_CODES[Ipopt.ApplicationReturnStatus(model.inner.status)]
    if status == MOI.UNKNOWN_RESULT_STATUS
        # Not sure. RestorationFailure can terminate at a feasible (but
        # non-stationary) point.
        return _manually_evaluated_primal_status(model)
    end
    return status
end

### MOI.DualStatus

function MOI.get(model::Optimizer, attr::MOI.DualStatus)
    if !(1 <= attr.result_index <= MOI.get(model, MOI.ResultCount()))
        return MOI.NO_SOLUTION
    end
    _, status = _STATUS_CODES[Ipopt.ApplicationReturnStatus(model.inner.status)]
    return status
end

### MOI.SolveTimeSec

MOI.get(model::Optimizer, ::MOI.SolveTimeSec) = model.solve_time

### MOI.BarrierIterations

MOI.get(model::Optimizer, ::MOI.BarrierIterations) = model.barrier_iterations

### MOI.ObjectiveValue

function MOI.get(model::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(model, attr)
    return _dual_multiplier(model) * model.inner.obj_val
end

### MOI.VariablePrimal

function MOI.get(
    model::Optimizer,
    attr::MOI.VariablePrimal,
    vi::MOI.VariableIndex,
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, vi)
    if MOI.Nonlinear._is_parameter(vi)
        ci = MOI.ConstraintIndex{MOI.VariableIndex,MOI.Parameter{Float64}}(
            vi.value,
        )
        return MOI.get(model.model, MOI.ConstraintSet(), ci).value
    end
    return model.inner.x[Ipopt.column(vi)]
end

### MOI.ConstraintPrimal

function row(model::Optimizer, ci::MOI.ConstraintIndex)
    return only(MOI.Nonlinear.constraint_rows(model.model, ci))
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{MOI.VectorOfVariables},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    f = MOI.get(model.model, MOI.ConstraintFunction(), ci)
    return MOI.get.(model, MOI.VariablePrimal(attr.result_index), f.variables)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{F,<:_SETS},
) where {
    F<:Union{
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
        MOI.ScalarNonlinearFunction,
    },
}
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

_dual_multiplier(model::Optimizer) = MOI.get(model.model, MOI.ObjectiveSense()) == MOI.MIN_SENSE ? 1.0 : -1.0

function MOI.get(
    model::Optimizer,
    attr::MOI.LagrangeMultiplier,
    ci::MOI.ConstraintIndex,
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    rows = MOI.Nonlinear.constraint_rows(model.model, ci)
    return -model.inner.mult_g[rows]
end

function MOI.supports(
    model::Optimizer,
    attr::MOI.LagrangeMultiplierStart,
    CI::Type{<:MOI.ConstraintIndex},
)
    return MOI.supports(model.model, attr, CI)
end
MOI.get(model::Optimizer, attr::MOI.LagrangeMultiplierStart, ci::MOI.ConstraintIndex) =
    MOI.get(model.model, attr, ci)
function MOI.set(
    model::Optimizer,
    attr::MOI.LagrangeMultiplierStart,
    ci::MOI.ConstraintIndex,
    value,
)
    return MOI.set(model.model, attr, ci, value)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VectorOfVariables},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    rows = Set(MOI.Nonlinear.constraint_rows(model.model, ci))
    structure = MOI.jacobian_structure(model.evaluator)
    values = zeros(length(structure))
    MOI.eval_constraint_jacobian(model.evaluator, values, model.inner.x)
    dual = zeros(MOI.get(model, MOI.NumberOfVariables()))
    sign = -1.0
    for ((r, c), value) in zip(structure, values)
        if r in rows
            dual[c] += sign * value * model.inner.mult_g[r]
        end
    end
    f = MOI.get(model.model, MOI.ConstraintFunction(), ci)
    return dual[getfield.(f.variables, :value)]
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{F,<:_SETS},
) where {
    F<:Union{
        MOI.ScalarAffineFunction{Float64},
        MOI.ScalarQuadraticFunction{Float64},
        MOI.ScalarNonlinearFunction,
    },
}
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    return -model.inner.mult_g[row(model, ci)]
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.LessThan{Float64}},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    rc = model.inner.mult_x_L[ci.value] - model.inner.mult_x_U[ci.value]
    return min(0.0, rc)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.GreaterThan{Float64}},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    rc = model.inner.mult_x_L[ci.value] - model.inner.mult_x_U[ci.value]
    return max(0.0, rc)
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.EqualTo{Float64}},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    rc = model.inner.mult_x_L[ci.value] - model.inner.mult_x_U[ci.value]
    return rc
end

function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{MOI.VariableIndex,MOI.Interval{Float64}},
)
    MOI.check_result_index_bounds(model, attr)
    MOI.throw_if_not_valid(model, ci)
    rc = model.inner.mult_x_L[ci.value] - model.inner.mult_x_U[ci.value]
    return rc
end

### MOI.NLPBlockDual

function MOI.get(model::Optimizer, attr::MOI.NLPBlockDual)
    MOI.check_result_index_bounds(model, attr)
    return -model.inner.mult_g[(length(model.inner.mult_g) - length(model.nlp_data.constraint_bounds) + 1):end]
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
    return model.inner.x[Ipopt.column(x)]
end
