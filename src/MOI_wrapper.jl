import MathOptInterface
const MOI = MathOptInterface

mutable struct VariableInfo
    lower_bound::Float64  # May be -Inf even if has_lower_bound == true
    has_lower_bound::Bool # Implies lower_bound == Inf
    upper_bound::Float64  # May be Inf even if has_upper_bound == true
    has_upper_bound::Bool # Implies upper_bound == Inf
    is_fixed::Bool        # Implies lower_bound == upper_bound and !has_lower_bound and !has_upper_bound.
    start::Float64
end
# The default start value is zero.
VariableInfo() = VariableInfo(-Inf, false, Inf, false, false, 0.0)

mutable struct Optimizer <: MOI.AbstractOptimizer
    inner::Union{IpoptProblem,Nothing}
    variable_info::Vector{VariableInfo}
    nlp_data::MOI.NLPBlockData
    sense::MOI.OptimizationSense
    objective::Union{MOI.SingleVariable,MOI.ScalarAffineFunction{Float64},MOI.ScalarQuadraticFunction{Float64},Nothing}
    linear_le_constraints::Vector{Tuple{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}}
    linear_ge_constraints::Vector{Tuple{MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}}}
    linear_eq_constraints::Vector{Tuple{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}}
    quadratic_le_constraints::Vector{Tuple{MOI.ScalarQuadraticFunction{Float64}, MOI.LessThan{Float64}}}
    quadratic_ge_constraints::Vector{Tuple{MOI.ScalarQuadraticFunction{Float64}, MOI.GreaterThan{Float64}}}
    quadratic_eq_constraints::Vector{Tuple{MOI.ScalarQuadraticFunction{Float64}, MOI.EqualTo{Float64}}}
    options
end

struct EmptyNLPEvaluator <: MOI.AbstractNLPEvaluator end
MOI.features_available(::EmptyNLPEvaluator) = [:Grad, :Jac, :Hess]
MOI.initialize(::EmptyNLPEvaluator, features) = nothing
MOI.eval_objective(::EmptyNLPEvaluator, x) = NaN
function MOI.eval_constraint(::EmptyNLPEvaluator, g, x)
    @assert length(g) == 0
    return
end
MOI.eval_objective_gradient(::EmptyNLPEvaluator, g, x) = nothing
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


Optimizer(;options...) = Optimizer(nothing, [], empty_nlp_data(), MOI.FEASIBILITY_SENSE, nothing, [], [], [], [], [], [], options)

MOI.supports(::Optimizer, ::MOI.NLPBlock) = true
MOI.supports(::Optimizer, ::MOI.ObjectiveFunction{MOI.SingleVariable}) = true
MOI.supports(::Optimizer, ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}) = true
MOI.supports(::Optimizer, ::MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}) = true
MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true
MOI.supports_constraint(::Optimizer, ::Type{MOI.SingleVariable}, ::Type{MOI.LessThan{Float64}}) = true
MOI.supports_constraint(::Optimizer, ::Type{MOI.SingleVariable}, ::Type{MOI.GreaterThan{Float64}}) = true
MOI.supports_constraint(::Optimizer, ::Type{MOI.SingleVariable}, ::Type{MOI.EqualTo{Float64}}) = true
MOI.supports_constraint(::Optimizer, ::Type{MOI.ScalarAffineFunction{Float64}}, ::Type{MOI.LessThan{Float64}}) = true
MOI.supports_constraint(::Optimizer, ::Type{MOI.ScalarAffineFunction{Float64}}, ::Type{MOI.GreaterThan{Float64}}) = true
MOI.supports_constraint(::Optimizer, ::Type{MOI.ScalarAffineFunction{Float64}}, ::Type{MOI.EqualTo{Float64}}) = true
MOI.supports_constraint(::Optimizer, ::Type{MOI.ScalarQuadraticFunction{Float64}}, ::Type{MOI.LessThan{Float64}}) = true
MOI.supports_constraint(::Optimizer, ::Type{MOI.ScalarQuadraticFunction{Float64}}, ::Type{MOI.GreaterThan{Float64}}) = true
MOI.supports_constraint(::Optimizer, ::Type{MOI.ScalarQuadraticFunction{Float64}}, ::Type{MOI.EqualTo{Float64}}) = true

function MOI.copy_to(model::Optimizer, src::MOI.ModelLike; copy_names = false)
    return MOI.Utilities.default_copy_to(model, src, copy_names)
end

MOI.get(model::Optimizer, ::MOI.NumberOfVariables) = length(model.variable_info)

function MOI.get(model::Optimizer, ::MOI.ListOfVariableIndices)
    return [MOI.VariableIndex(i) for i in 1:length(model.variable_info)]
end


function MOI.set(model::Optimizer, ::MOI.ObjectiveSense,
                 sense::MOI.OptimizationSense)
    model.sense = sense
    return
end

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

function check_inbounds(model::Optimizer, vi::MOI.VariableIndex)
    num_variables = length(model.variable_info)
    if !(1 <= vi.value <= num_variables)
        error("Invalid variable index $vi. ($num_variables variables in the model.)")
    end
end

function check_inbounds(model::Optimizer, var::MOI.SingleVariable)
    return check_inbounds(model, var.variable)
end

function check_inbounds(model::Optimizer, aff::MOI.ScalarAffineFunction)
    for term in aff.terms
        check_inbounds(model, term.variable_index)
    end
end

function check_inbounds(model::Optimizer, quad::MOI.ScalarQuadraticFunction)
    for term in quad.affine_terms
        check_inbounds(model, term.variable_index)
    end
    for term in quad.quadratic_terms
        check_inbounds(model, term.variable_index_1)
        check_inbounds(model, term.variable_index_2)
    end
end

function has_upper_bound(model::Optimizer, vi::MOI.VariableIndex)
    return model.variable_info[vi.value].has_upper_bound
end

function has_lower_bound(model::Optimizer, vi::MOI.VariableIndex)
    return model.variable_info[vi.value].has_lower_bound
end

function is_fixed(model::Optimizer, vi::MOI.VariableIndex)
    return model.variable_info[vi.value].is_fixed
end

function MOI.add_constraint(model::Optimizer, v::MOI.SingleVariable, lt::MOI.LessThan{Float64})
    vi = v.variable
    check_inbounds(model, vi)
    if isnan(lt.upper)
        error("Invalid upper bound value $(lt.upper).")
    end
    if has_upper_bound(model, vi)
        error("Upper bound on variable $vi already exists.")
    end
    if is_fixed(model, vi)
        error("Variable $vi is fixed. Cannot also set upper bound.")
    end
    model.variable_info[vi.value].upper_bound = lt.upper
    model.variable_info[vi.value].has_upper_bound = true
    return MOI.ConstraintIndex{MOI.SingleVariable, MOI.LessThan{Float64}}(vi.value)
end

function MOI.add_constraint(model::Optimizer, v::MOI.SingleVariable, gt::MOI.GreaterThan{Float64})
    vi = v.variable
    check_inbounds(model, vi)
    if isnan(gt.lower)
        error("Invalid lower bound value $(gt.lower).")
    end
    if has_lower_bound(model, vi)
        error("Lower bound on variable $vi already exists.")
    end
    if is_fixed(model, vi)
        error("Variable $vi is fixed. Cannot also set lower bound.")
    end
    model.variable_info[vi.value].lower_bound = gt.lower
    model.variable_info[vi.value].has_lower_bound = true
    return MOI.ConstraintIndex{MOI.SingleVariable, MOI.GreaterThan{Float64}}(vi.value)
end

function MOI.add_constraint(model::Optimizer, v::MOI.SingleVariable, eq::MOI.EqualTo{Float64})
    vi = v.variable
    check_inbounds(model, vi)
    if isnan(eq.value)
        error("Invalid fixed value $(gt.lower).")
    end
    if has_lower_bound(model, vi)
        error("Variable $vi has a lower bound. Cannot be fixed.")
    end
    if has_upper_bound(model, vi)
        error("Variable $vi has an upper bound. Cannot be fixed.")
    end
    if is_fixed(model, vi)
        error("Variable $vi is already fixed.")
    end
    model.variable_info[vi.value].lower_bound = eq.value
    model.variable_info[vi.value].upper_bound = eq.value
    model.variable_info[vi.value].is_fixed = true
    return MOI.ConstraintIndex{MOI.SingleVariable, MOI.EqualTo{Float64}}(vi.value)
end

macro define_add_constraint(function_type, set_type, array_name)
    quote
        function MOI.add_constraint(model::Optimizer, func::$function_type, set::$set_type)
            check_inbounds(model, func)
            push!(model.$(array_name), (func, set))
            return MOI.ConstraintIndex{$function_type, $set_type}(length(model.$(array_name)))
        end
    end
end

@define_add_constraint(MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64},
                       linear_le_constraints)
@define_add_constraint(MOI.ScalarAffineFunction{Float64},
                       MOI.GreaterThan{Float64}, linear_ge_constraints)
@define_add_constraint(MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64},
                       linear_eq_constraints)
@define_add_constraint(MOI.ScalarQuadraticFunction{Float64},
                       MOI.LessThan{Float64}, quadratic_le_constraints)
@define_add_constraint(MOI.ScalarQuadraticFunction{Float64},
                       MOI.GreaterThan{Float64}, quadratic_ge_constraints)
@define_add_constraint(MOI.ScalarQuadraticFunction{Float64},
                       MOI.EqualTo{Float64}, quadratic_eq_constraints)

function MOI.supports(::Optimizer, ::MOI.VariablePrimalStart,
                      ::Type{MOI.VariableIndex})
    return true
end
function MOI.set(model::Optimizer, ::MOI.VariablePrimalStart,
                 vi::MOI.VariableIndex, value::Real)
    check_inbounds(model, vi)
    model.variable_info[vi.value].start = value
    return
end

function MOI.set(model::Optimizer, ::MOI.NLPBlock, nlp_data::MOI.NLPBlockData)
    model.nlp_data = nlp_data
    return
end

function MOI.set(model::Optimizer, ::MOI.ObjectiveFunction,
                 func::Union{MOI.SingleVariable, MOI.ScalarAffineFunction,
                             MOI.ScalarQuadraticFunction})
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
linear_eq_offset(model::Optimizer) = linear_ge_offset(model) + length(model.linear_ge_constraints)
quadratic_le_offset(model::Optimizer) = linear_eq_offset(model) + length(model.linear_eq_constraints)
quadratic_ge_offset(model::Optimizer) = quadratic_le_offset(model) + length(model.quadratic_le_constraints)
quadratic_eq_offset(model::Optimizer) = quadratic_ge_offset(model) + length(model.quadratic_ge_constraints)
nlp_constraint_offset(model::Optimizer) = quadratic_eq_offset(model) + length(model.quadratic_eq_constraints)

# Convenience functions used only in optimize!

function append_to_jacobian_sparsity!(jacobian_sparsity, aff::MOI.ScalarAffineFunction, row)
    for term in aff.terms
        push!(jacobian_sparsity, (row, term.variable_index.value))
    end
end

function append_to_jacobian_sparsity!(jacobian_sparsity, quad::MOI.ScalarQuadraticFunction, row)
    for term in quad.affine_terms
        push!(jacobian_sparsity, (row, term.variable_index.value))
    end
    for term in quad.quadratic_terms
        row_idx = term.variable_index_1
        col_idx = term.variable_index_2
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
        for (func, set) in $(esc(array_name))
            append_to_jacobian_sparsity!($(esc(:jacobian_sparsity)), func, $escrow)
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

append_to_hessian_sparsity!(hessian_sparsity, ::Union{MOI.SingleVariable,MOI.ScalarAffineFunction}) = nothing

function append_to_hessian_sparsity!(hessian_sparsity, quad::MOI.ScalarQuadraticFunction)
    for term in quad.quadratic_terms
        push!(hessian_sparsity, (term.variable_index_1.value,
                                 term.variable_index_2.value))
    end
end

function hessian_lagrangian_structure(model::Optimizer)
    hessian_sparsity = Tuple{Int64,Int64}[]
    if !model.nlp_data.has_objective && model.objective !== nothing
        append_to_hessian_sparsity!(hessian_sparsity, model.objective)
    end
    for (quad, set) in model.quadratic_le_constraints
        append_to_hessian_sparsity!(hessian_sparsity, quad)
    end
    for (quad, set) in model.quadratic_ge_constraints
        append_to_hessian_sparsity!(hessian_sparsity, quad)
    end
    for (quad, set) in model.quadratic_eq_constraints
        append_to_hessian_sparsity!(hessian_sparsity, quad)
    end
    nlp_hessian_sparsity = MOI.hessian_lagrangian_structure(model.nlp_data.evaluator)
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
        function_value += term.coefficient*x[term.variable_index.value]
    end
    return function_value
end

function eval_function(quad::MOI.ScalarQuadraticFunction, x)
    function_value = quad.constant
    for term in quad.affine_terms
        function_value += term.coefficient*x[term.variable_index.value]
    end
    for term in quad.quadratic_terms
        row_idx = term.variable_index_1
        col_idx = term.variable_index_2
        coefficient = term.coefficient
        if row_idx == col_idx
            function_value += 0.5*coefficient*x[row_idx.value]*x[col_idx.value]
        else
            function_value += coefficient*x[row_idx.value]*x[col_idx.value]
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
    grad[var.variable.value] = 1.0
end

function fill_gradient!(grad, x, aff::MOI.ScalarAffineFunction{Float64})
    fill!(grad, 0.0)
    for term in aff.terms
        grad[term.variable_index.value] += term.coefficient
    end
end

function fill_gradient!(grad, x, quad::MOI.ScalarQuadraticFunction{Float64})
    fill!(grad, 0.0)
    for term in quad.affine_terms
        grad[term.variable_index.value] += term.coefficient
    end
    for term in quad.quadratic_terms
        row_idx = term.variable_index_1
        col_idx = term.variable_index_2
        coefficient = term.coefficient
        if row_idx == col_idx
            grad[row_idx.value] += coefficient*x[row_idx.value]
        else
            grad[row_idx.value] += coefficient*x[col_idx.value]
            grad[col_idx.value] += coefficient*x[row_idx.value]
        end
    end
end

function eval_objective_gradient(model::Optimizer, grad, x)
    if model.nlp_data.has_objective
        MOI.eval_objective_gradient(model.nlp_data.evaluator, grad, x)
    elseif model.objective !== nothing
        fill_gradient!(grad, x, model.objective)
    end
    return
end

# Refers to local variables in eval_constraint() below.
macro eval_function(array_name)
    escrow = esc(:row)
    quote
        for (func, set) in $(esc(array_name))
            $(esc(:g))[$escrow] = eval_function(func, $(esc(:x)))
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

function fill_constraint_jacobian!(values, start_offset, x, aff::MOI.ScalarAffineFunction)
    num_coefficients = length(aff.terms)
    for i in 1:num_coefficients
        values[start_offset+i] = aff.terms[i].coefficient
    end
    return num_coefficients
end

function fill_constraint_jacobian!(values, start_offset, x, quad::MOI.ScalarQuadraticFunction)
    num_affine_coefficients = length(quad.affine_terms)
    for i in 1:num_affine_coefficients
        values[start_offset+i] = quad.affine_terms[i].coefficient
    end
    num_quadratic_coefficients = 0
    for term in quad.quadratic_terms
        row_idx = term.variable_index_1
        col_idx = term.variable_index_2
        coefficient = term.coefficient
        if row_idx == col_idx
            values[start_offset+num_affine_coefficients+num_quadratic_coefficients+1] = coefficient*x[col_idx.value]
            num_quadratic_coefficients += 1
        else
            # Note that the order matches the Jacobian sparsity pattern.
            values[start_offset+num_affine_coefficients+num_quadratic_coefficients+1] = coefficient*x[col_idx.value]
            values[start_offset+num_affine_coefficients+num_quadratic_coefficients+2] = coefficient*x[row_idx.value]
            num_quadratic_coefficients += 2
        end
    end
    return num_affine_coefficients + num_quadratic_coefficients
end

# Refers to local variables in eval_constraint_jacobian() below.
macro fill_constraint_jacobian(array_name)
    esc_offset = esc(:offset)
    quote
        for (func, set) in $(esc(array_name))
            $esc_offset += fill_constraint_jacobian!($(esc(:values)),
                                                     $esc_offset, $(esc(:x)),
                                                     func)
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

function fill_hessian_lagrangian!(values, start_offset, scale_factor,
                                  ::Union{MOI.SingleVariable,
                                          MOI.ScalarAffineFunction,Nothing})
    return 0
end

function fill_hessian_lagrangian!(values, start_offset, scale_factor,
                                  quad::MOI.ScalarQuadraticFunction)
    for i in 1:length(quad.quadratic_terms)
        values[start_offset + i] = scale_factor*quad.quadratic_terms[i].coefficient
    end
    return length(quad.quadratic_terms)
end

function eval_hessian_lagrangian(model::Optimizer, values, x, obj_factor, lambda)
    offset = 0
    if !model.nlp_data.has_objective
        offset += fill_hessian_lagrangian!(values, 0, obj_factor,
                                          model.objective)
    end
    for (i, (quad, set)) in enumerate(model.quadratic_le_constraints)
        offset += fill_hessian_lagrangian!(values, offset, lambda[i+quadratic_le_offset(model)], quad)
    end
    for (i, (quad, set)) in enumerate(model.quadratic_ge_constraints)
        offset += fill_hessian_lagrangian!(values, offset, lambda[i+quadratic_ge_offset(model)], quad)
    end
    for (i, (quad, set)) in enumerate(model.quadratic_eq_constraints)
        offset += fill_hessian_lagrangian!(values, offset, lambda[i+quadratic_eq_offset(model)], quad)
    end
    nlp_values = view(values, 1 + offset : length(values))
    nlp_lambda = view(lambda, 1 + nlp_constraint_offset(model) : length(lambda))
    MOI.eval_hessian_lagrangian(model.nlp_data.evaluator, nlp_values, x, obj_factor, nlp_lambda)
end

function constraint_bounds(model::Optimizer)
    constraint_lb = Float64[]
    constraint_ub = Float64[]
    for (func, set) in model.linear_le_constraints
        push!(constraint_lb, -Inf)
        push!(constraint_ub, set.upper)
    end
    for (func, set) in model.linear_ge_constraints
        push!(constraint_lb, set.lower)
        push!(constraint_ub, Inf)
    end
    for (func, set) in model.linear_eq_constraints
        push!(constraint_lb, set.value)
        push!(constraint_ub, set.value)
    end
    for (func, set) in model.quadratic_le_constraints
        push!(constraint_lb, -Inf)
        push!(constraint_ub, set.upper)
    end
    for (func, set) in model.quadratic_ge_constraints
        push!(constraint_lb, set.lower)
        push!(constraint_ub, Inf)
    end
    for (func, set) in model.quadratic_eq_constraints
        push!(constraint_lb, set.value)
        push!(constraint_ub, set.value)
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
    num_quadratic_constraints = nlp_constraint_offset(model) -
                                quadratic_le_offset(model)
    num_nlp_constraints = length(model.nlp_data.constraint_bounds)
    num_constraints = num_nlp_constraints + nlp_row_offset

    evaluator = model.nlp_data.evaluator
    features = MOI.features_available(evaluator)
    has_hessian = (:Hess in features)
    init_feat = [:Grad]
    has_hessian && push!(init_feat, :Hess)
    num_nlp_constraints > 0 && push!(init_feat, :Jac)

    MOI.initialize(evaluator, init_feat)
    jacobian_sparsity = jacobian_structure(model)
    hessian_sparsity = has_hessian ? hessian_lagrangian_structure(model) : []

    # Objective callback
    if model.sense == MOI.MIN_SENSE
        objective_scale = 1.0
    elseif model.sense == MOI.MAX_SENSE
        objective_scale = -1.0
    else # FEASIBILITY_SENSE
        # TODO: This could produce confusing solver output if a nonzero
        # objective is set.
        objective_scale = 0.0
    end

    eval_f_cb(x) = objective_scale * eval_objective(model, x)

    # Objective gradient callback
    function eval_grad_f_cb(x, grad_f)
        eval_objective_gradient(model, grad_f, x)
        Compat.rmul!(grad_f,objective_scale)
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
    end

    if has_hessian
        # Hessian callback
        function eval_h_cb(x, mode, rows, cols, obj_factor,
            lambda, values)
            if mode == :Structure
                for i in 1:length(hessian_sparsity)
                    rows[i] = hessian_sparsity[i][1]
                    cols[i] = hessian_sparsity[i][2]
                end
            else
                obj_factor *= objective_scale
                eval_hessian_lagrangian(model, values, x, objective_scale*obj_factor, lambda)
            end
        end
    else
        eval_h_cb = nothing
    end

    x_l = [v.lower_bound for v in model.variable_info]
    x_u = [v.upper_bound for v in model.variable_info]

    constraint_lb, constraint_ub = constraint_bounds(model)

    model.inner = createProblem(num_variables, x_l, x_u, num_constraints,
                            constraint_lb, constraint_ub,
                            length(jacobian_sparsity),
                            length(hessian_sparsity),
                            eval_f_cb, eval_g_cb, eval_grad_f_cb, eval_jac_g_cb,
                            eval_h_cb)

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

    model.inner.x = [v.start for v in model.variable_info]

    for (name,value) in model.options
        sname = string(name)
        if match(r"(^resto_)", sname) != nothing
            sname = replace(sname, r"(^resto_)", "resto.")
        end
        addOption(model.inner, sname, value)
    end
    solveProblem(model.inner)
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
        return MOI.OHTER_ERROR
    elseif status == :Insufficient_Memory
        return MOI.MEMORY_LIMIT
    else
        error("Unrecognized Ipopt status $status")
    end
end

# Ipopt always has an iterate available.
function MOI.get(model::Optimizer, ::MOI.ResultCount)
    return (model.inner !== nothing) ? 1 : 0
end

function MOI.get(model::Optimizer, ::MOI.PrimalStatus)
    if model.inner === nothing
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

function MOI.get(model::Optimizer, ::MOI.DualStatus)
    if model.inner === nothing
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

function MOI.get(model::Optimizer, ::MOI.ObjectiveValue)
    if model.inner === nothing
        error("ObjectiveValue not available.")
    end
    scale = (model.sense == MOI.MAX_SENSE) ? -1 : 1
    return scale * model.inner.obj_val
end

# TODO: This is a bit off, because the variable primal should be available
# only after a solve. If model.inner is initialized but we haven't solved, then
# the primal values we return do not have the intended meaning.
function MOI.get(model::Optimizer, ::MOI.VariablePrimal, vi::MOI.VariableIndex)
    if model.inner === nothing
        error("VariablePrimal not available.")
    end
    check_inbounds(model, vi)
    return model.inner.x[vi.value]
end

macro define_constraint_primal(function_type, set_type, prefix)
    constraint_array = Symbol(string(prefix) * "_constraints")
    offset_function = Symbol(string(prefix) * "_offset")
    quote
        function MOI.get(model::Optimizer, ::MOI.ConstraintPrimal,
                         ci::MOI.ConstraintIndex{$function_type, $set_type})
            if model.inner === nothing
                error("ConstraintPrimal not available.")
            end
            if !(1 <= ci.value <= length(model.$(constraint_array)))
                error("Invalid constraint index ", ci.value)
            end
            return model.inner.g[ci.value + $offset_function(model)]
        end
    end
end

@define_constraint_primal(MOI.ScalarAffineFunction{Float64},
                          MOI.LessThan{Float64}, linear_le)
@define_constraint_primal(MOI.ScalarAffineFunction{Float64},
                          MOI.GreaterThan{Float64}, linear_ge)
@define_constraint_primal(MOI.ScalarAffineFunction{Float64},
                          MOI.EqualTo{Float64}, linear_eq)
@define_constraint_primal(MOI.ScalarQuadraticFunction{Float64},
                          MOI.LessThan{Float64}, quadratic_le)
@define_constraint_primal(MOI.ScalarQuadraticFunction{Float64},
                          MOI.GreaterThan{Float64}, quadratic_ge)
@define_constraint_primal(MOI.ScalarQuadraticFunction{Float64},
                          MOI.EqualTo{Float64}, quadratic_eq)

function MOI.get(model::Optimizer, ::MOI.ConstraintPrimal,
                 ci::MOI.ConstraintIndex{MOI.SingleVariable,
                                         MOI.LessThan{Float64}})
    if model.inner === nothing
        error("ConstraintPrimal not available.")
    end
    vi = MOI.VariableIndex(ci.value)
    check_inbounds(model, vi)
    if !has_upper_bound(model, vi)
        error("Variable $vi has no upper bound -- ConstraintPrimal not defined.")
    end
    return model.inner.x[vi.value]
end

function MOI.get(model::Optimizer, ::MOI.ConstraintPrimal,
                 ci::MOI.ConstraintIndex{MOI.SingleVariable,
                                         MOI.GreaterThan{Float64}})
    if model.inner === nothing
        error("ConstraintPrimal not available.")
    end
    vi = MOI.VariableIndex(ci.value)
    check_inbounds(model, vi)
    if !has_lower_bound(model, vi)
        error("Variable $vi has no lower bound -- ConstraintPrimal not defined.")
    end
    return model.inner.x[vi.value]
end

function MOI.get(model::Optimizer, ::MOI.ConstraintPrimal,
                 ci::MOI.ConstraintIndex{MOI.SingleVariable,
                                         MOI.EqualTo{Float64}})
    if model.inner === nothing
        error("ConstraintPrimal not available.")
    end
    vi = MOI.VariableIndex(ci.value)
    check_inbounds(model, vi)
    if !is_fixed(model, vi)
        error("Variable $vi is not fixed -- ConstraintPrimal not defined.")
    end
    return model.inner.x[vi.value]
end

function MOI.get(model::Optimizer, ::MOI.ConstraintDual,
                 ci::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},
                                         MOI.LessThan{Float64}})
    if model.inner === nothing
        error("ConstraintDual not available.")
    end
    @assert 1 <= ci.value <= length(model.linear_le_constraints)
    # TODO: Unable to find documentation in Ipopt about the signs of duals.
    # Rescaling by -1 here seems to pass the MOI tests.
    return -1 * model.inner.mult_g[ci.value + linear_le_offset(model)]
end

function MOI.get(model::Optimizer, ::MOI.ConstraintDual,
                 ci::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},
                                         MOI.GreaterThan{Float64}})
    if model.inner === nothing
        error("ConstraintDual not available.")
    end
    @assert 1 <= ci.value <= length(model.linear_ge_constraints)
    # TODO: Unable to find documentation in Ipopt about the signs of duals.
    # Rescaling by -1 here seems to pass the MOI tests.
    return -1 * model.inner.mult_g[ci.value + linear_ge_offset(model)]
end

function MOI.get(model::Optimizer, ::MOI.ConstraintDual,
                 ci::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},
                                         MOI.EqualTo{Float64}})
    if model.inner === nothing
        error("ConstraintDual not available.")
    end
    @assert 1 <= ci.value <= length(model.linear_eq_constraints)
    # TODO: Rescaling by -1 for consistency, but I don't know if this is covered
    # by tests.
    return -1 * model.inner.mult_g[ci.value + linear_eq_offset(model)]
end

function MOI.get(model::Optimizer, ::MOI.ConstraintDual,
                 ci::MOI.ConstraintIndex{MOI.SingleVariable,
                                         MOI.LessThan{Float64}})
    if model.inner === nothing
        error("ConstraintDual not available.")
    end
    vi = MOI.VariableIndex(ci.value)
    check_inbounds(model, vi)
    if !has_upper_bound(model, vi)
        error("Variable $vi has no upper bound -- ConstraintDual not defined.")
    end
    # MOI convention is for feasible LessThan duals to be nonpositive.
    return -1 * model.inner.mult_x_U[vi.value]
end

function MOI.get(model::Optimizer, ::MOI.ConstraintDual,
                 ci::MOI.ConstraintIndex{MOI.SingleVariable,
                                         MOI.GreaterThan{Float64}})
    if model.inner === nothing
        error("ConstraintDual not available.")
    end
    vi = MOI.VariableIndex(ci.value)
    check_inbounds(model, vi)
    if !has_lower_bound(model, vi)
        error("Variable $vi has no lower bound -- ConstraintDual not defined.")
    end
    return model.inner.mult_x_L[vi.value]
end

function MOI.get(model::Optimizer, ::MOI.ConstraintDual,
                 ci::MOI.ConstraintIndex{MOI.SingleVariable,
                                         MOI.EqualTo{Float64}})
    if model.inner === nothing
        error("ConstraintDual not available.")
    end
    vi = MOI.VariableIndex(ci.value)
    check_inbounds(model, vi)
    if !is_fixed(model, vi)
        error("Variable $vi is not fixed -- ConstraintDual not defined.")
    end
    return model.inner.mult_x_L[vi.value] - model.inner.mult_x_U[vi.value]
end

function MOI.get(model::Optimizer, ::MOI.NLPBlockDual)
    if model.inner === nothing
        error("NLPBlockDual not available.")
    end
    return -1 * model.inner.mult_g[(1 + nlp_constraint_offset(model)):end]
end
