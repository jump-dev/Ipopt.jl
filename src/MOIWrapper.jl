using MathOptInterface
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

export IpoptOptimizer
mutable struct IpoptOptimizer <: MOI.AbstractOptimizer
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
MOI.initialize!(::EmptyNLPEvaluator, features) = nothing
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


IpoptOptimizer(;options...) = IpoptOptimizer(nothing, [], empty_nlp_data(), MOI.FeasibilitySense, nothing, [], [], [], [], [], [], options)

MOI.supports(::IpoptOptimizer, ::MOI.NLPBlock) = true
MOI.supports(::IpoptOptimizer, ::MOI.ObjectiveFunction{MOI.SingleVariable}) = true
MOI.supports(::IpoptOptimizer, ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}) = true
MOI.supports(::IpoptOptimizer, ::MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}) = true
MOI.supports(::IpoptOptimizer, ::MOI.ObjectiveSense) = true
MOI.supportsconstraint(::IpoptOptimizer, ::Type{MOI.SingleVariable}, ::Type{MOI.LessThan{Float64}}) = true
MOI.supportsconstraint(::IpoptOptimizer, ::Type{MOI.SingleVariable}, ::Type{MOI.GreaterThan{Float64}}) = true
MOI.supportsconstraint(::IpoptOptimizer, ::Type{MOI.SingleVariable}, ::Type{MOI.EqualTo{Float64}}) = true
MOI.supportsconstraint(::IpoptOptimizer, ::Type{MOI.ScalarAffineFunction{Float64}}, ::Type{MOI.LessThan{Float64}}) = true
MOI.supportsconstraint(::IpoptOptimizer, ::Type{MOI.ScalarAffineFunction{Float64}}, ::Type{MOI.GreaterThan{Float64}}) = true
MOI.supportsconstraint(::IpoptOptimizer, ::Type{MOI.ScalarAffineFunction{Float64}}, ::Type{MOI.EqualTo{Float64}}) = true
MOI.supportsconstraint(::IpoptOptimizer, ::Type{MOI.ScalarQuadraticFunction{Float64}}, ::Type{MOI.LessThan{Float64}}) = true
MOI.supportsconstraint(::IpoptOptimizer, ::Type{MOI.ScalarQuadraticFunction{Float64}}, ::Type{MOI.GreaterThan{Float64}}) = true
MOI.supportsconstraint(::IpoptOptimizer, ::Type{MOI.ScalarQuadraticFunction{Float64}}, ::Type{MOI.EqualTo{Float64}}) = true

MOI.canaddvariable(::IpoptOptimizer) = true
# TODO: The distinction between supportsconstraint and canaddconstraint is maybe too subtle.
MOI.canaddconstraint(::IpoptOptimizer, ::Type{MOI.SingleVariable}, ::Type{MOI.LessThan{Float64}}) = true
MOI.canaddconstraint(::IpoptOptimizer, ::Type{MOI.SingleVariable}, ::Type{MOI.GreaterThan{Float64}}) = true
MOI.canaddconstraint(::IpoptOptimizer, ::Type{MOI.SingleVariable}, ::Type{MOI.EqualTo{Float64}}) = true
MOI.canaddconstraint(::IpoptOptimizer, ::Type{MOI.ScalarAffineFunction{Float64}}, ::Type{MOI.LessThan{Float64}}) = true
MOI.canaddconstraint(::IpoptOptimizer, ::Type{MOI.ScalarAffineFunction{Float64}}, ::Type{MOI.GreaterThan{Float64}}) = true
MOI.canaddconstraint(::IpoptOptimizer, ::Type{MOI.ScalarAffineFunction{Float64}}, ::Type{MOI.EqualTo{Float64}}) = true
MOI.canaddconstraint(::IpoptOptimizer, ::Type{MOI.ScalarQuadraticFunction{Float64}}, ::Type{MOI.LessThan{Float64}}) = true
MOI.canaddconstraint(::IpoptOptimizer, ::Type{MOI.ScalarQuadraticFunction{Float64}}, ::Type{MOI.GreaterThan{Float64}}) = true
MOI.canaddconstraint(::IpoptOptimizer, ::Type{MOI.ScalarQuadraticFunction{Float64}}, ::Type{MOI.EqualTo{Float64}}) = true

MOI.canset(::IpoptOptimizer, ::MOI.VariablePrimalStart, ::Type{MOI.VariableIndex}) = true
MOI.canset(::IpoptOptimizer, ::MOI.ObjectiveSense) = true
MOI.canset(::IpoptOptimizer, ::MOI.NLPBlock) = true
MOI.canset(::IpoptOptimizer, ::MOI.ObjectiveFunction{MOI.SingleVariable}) = true
MOI.canset(::IpoptOptimizer, ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}) = true
MOI.canset(::IpoptOptimizer, ::MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}) = true

MOI.copy!(m::IpoptOptimizer, src::MOI.ModelLike; copynames = false) = MOI.Utilities.defaultcopy!(m, src, copynames)

MOI.canget(::IpoptOptimizer, ::MOI.NumberOfVariables) = true
MOI.get(m::IpoptOptimizer, ::MOI.NumberOfVariables) = length(m.variable_info)

MOI.canget(::IpoptOptimizer, ::MOI.ListOfVariableIndices) = true
function MOI.get(m::IpoptOptimizer, ::MOI.ListOfVariableIndices)
    return [MOI.VariableIndex(i) for i in 1:length(m.variable_info)]
end


function MOI.set!(m::IpoptOptimizer, ::MOI.ObjectiveSense, sense::MOI.OptimizationSense)
    m.sense = sense
    return
end

function MOI.empty!(m::IpoptOptimizer)
    m.inner = nothing
    empty!(m.variable_info)
    m.nlp_data = empty_nlp_data()
    m.sense = MOI.FeasibilitySense
    m.objective = nothing
    empty!(m.linear_le_constraints)
    empty!(m.linear_ge_constraints)
    empty!(m.linear_eq_constraints)
    empty!(m.quadratic_le_constraints)
    empty!(m.quadratic_ge_constraints)
    empty!(m.quadratic_eq_constraints)
end

function MOI.isempty(m::IpoptOptimizer)
    return isempty(m.variable_info) &&
           m.nlp_data.evaluator isa EmptyNLPEvaluator &&
           m.sense == MOI.FeasibilitySense &&
           isempty(m.linear_le_constraints) &&
           isempty(m.linear_ge_constraints) &&
           isempty(m.linear_eq_constraints) &&
           isempty(m.quadratic_le_constraints) &&
           isempty(m.quadratic_ge_constraints) &&
           isempty(m.quadratic_eq_constraints)
end

function MOI.addvariable!(m::IpoptOptimizer)
    push!(m.variable_info, VariableInfo())
    return MOI.VariableIndex(length(m.variable_info))
end
MOI.addvariables!(m::IpoptOptimizer, n::Int) = [MOI.addvariable!(m) for i in 1:n]

function check_inbounds(m::IpoptOptimizer, vi::MOI.VariableIndex)
    num_variables = length(m.variable_info)
    if !(1 <= vi.value <= num_variables)
        error("Invalid variable index $vi. ($num_variables variables in the model.)")
    end
end

check_inbounds(m::IpoptOptimizer, var::MOI.SingleVariable) = check_inbounds(m, var.variable)

function check_inbounds(m::IpoptOptimizer, aff::MOI.ScalarAffineFunction)
    for term in aff.terms
        check_inbounds(m, term.variable_index)
    end
end

function check_inbounds(m::IpoptOptimizer, quad::MOI.ScalarQuadraticFunction)
    for term in quad.affine_terms
        check_inbounds(m, term.variable_index)
    end
    for term in quad.quadratic_terms
        check_inbounds(m, term.variable_index_1)
        check_inbounds(m, term.variable_index_2)
    end
end

function has_upper_bound(m::IpoptOptimizer, vi::MOI.VariableIndex)
    return m.variable_info[vi.value].has_upper_bound
end

function has_lower_bound(m::IpoptOptimizer, vi::MOI.VariableIndex)
    return m.variable_info[vi.value].has_lower_bound
end

function is_fixed(m::IpoptOptimizer, vi::MOI.VariableIndex)
    return m.variable_info[vi.value].is_fixed
end

function MOI.addconstraint!(m::IpoptOptimizer, v::MOI.SingleVariable, lt::MOI.LessThan{Float64})
    vi = v.variable
    check_inbounds(m, vi)
    if isnan(lt.upper)
        error("Invalid upper bound value $(lt.upper).")
    end
    if has_upper_bound(m, vi)
        error("Upper bound on variable $vi already exists.")
    end
    if is_fixed(m, vi)
        error("Variable $vi is fixed. Cannot also set upper bound.")
    end
    m.variable_info[vi.value].upper_bound = lt.upper
    m.variable_info[vi.value].has_upper_bound = true
    return MOI.ConstraintIndex{MOI.SingleVariable, MOI.LessThan{Float64}}(vi.value)
end

function MOI.addconstraint!(m::IpoptOptimizer, v::MOI.SingleVariable, gt::MOI.GreaterThan{Float64})
    vi = v.variable
    check_inbounds(m, vi)
    if isnan(gt.lower)
        error("Invalid lower bound value $(gt.lower).")
    end
    if has_lower_bound(m, vi)
        error("Lower bound on variable $vi already exists.")
    end
    if is_fixed(m, vi)
        error("Variable $vi is fixed. Cannot also set lower bound.")
    end
    m.variable_info[vi.value].lower_bound = gt.lower
    m.variable_info[vi.value].has_lower_bound = true
    return MOI.ConstraintIndex{MOI.SingleVariable, MOI.GreaterThan{Float64}}(vi.value)
end

function MOI.addconstraint!(m::IpoptOptimizer, v::MOI.SingleVariable, eq::MOI.EqualTo{Float64})
    vi = v.variable
    check_inbounds(m, vi)
    if isnan(eq.value)
        error("Invalid fixed value $(gt.lower).")
    end
    if has_lower_bound(m, vi)
        error("Variable $vi has a lower bound. Cannot be fixed.")
    end
    if has_upper_bound(m, vi)
        error("Variable $vi has an upper bound. Cannot be fixed.")
    end
    if is_fixed(m, vi)
        error("Variable $vi is already fixed.")
    end
    m.variable_info[vi.value].lower_bound = eq.value
    m.variable_info[vi.value].upper_bound = eq.value
    m.variable_info[vi.value].is_fixed = true
    return MOI.ConstraintIndex{MOI.SingleVariable, MOI.EqualTo{Float64}}(vi.value)
end

macro define_addconstraint(function_type, set_type, array_name)
    quote
        function MOI.addconstraint!(m::IpoptOptimizer, func::$function_type, set::$set_type)
            check_inbounds(m, func)
            push!(m.$(array_name), (func, set))
            return MOI.ConstraintIndex{$function_type, $set_type}(length(m.$(array_name)))
        end
    end
end

@define_addconstraint MOI.ScalarAffineFunction{Float64} MOI.LessThan{Float64} linear_le_constraints
@define_addconstraint MOI.ScalarAffineFunction{Float64} MOI.GreaterThan{Float64} linear_ge_constraints
@define_addconstraint MOI.ScalarAffineFunction{Float64} MOI.EqualTo{Float64} linear_eq_constraints
@define_addconstraint MOI.ScalarQuadraticFunction{Float64} MOI.LessThan{Float64} quadratic_le_constraints
@define_addconstraint MOI.ScalarQuadraticFunction{Float64} MOI.GreaterThan{Float64} quadratic_ge_constraints
@define_addconstraint MOI.ScalarQuadraticFunction{Float64} MOI.EqualTo{Float64} quadratic_eq_constraints

function MOI.set!(m::IpoptOptimizer, ::MOI.VariablePrimalStart, vi::MOI.VariableIndex, value::Real)
    check_inbounds(m, vi)
    m.variable_info[vi.value].start = value
    return
end

function MOI.set!(m::IpoptOptimizer, ::MOI.NLPBlock, nlp_data::MOI.NLPBlockData)
    m.nlp_data = nlp_data
    return
end

function MOI.set!(m::IpoptOptimizer, ::MOI.ObjectiveFunction, func::Union{MOI.SingleVariable,MOI.ScalarAffineFunction,MOI.ScalarQuadraticFunction})
    check_inbounds(m, func)
    m.objective = func
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

linear_le_offset(m::IpoptOptimizer) = 0
linear_ge_offset(m::IpoptOptimizer) = length(m.linear_le_constraints)
linear_eq_offset(m::IpoptOptimizer) = linear_ge_offset(m) + length(m.linear_ge_constraints)
quadratic_le_offset(m::IpoptOptimizer) = linear_eq_offset(m) + length(m.linear_eq_constraints)
quadratic_ge_offset(m::IpoptOptimizer) = quadratic_le_offset(m) + length(m.quadratic_le_constraints)
quadratic_eq_offset(m::IpoptOptimizer) = quadratic_ge_offset(m) + length(m.quadratic_ge_constraints)
nlp_constraint_offset(m::IpoptOptimizer) = quadratic_eq_offset(m) + length(m.quadratic_eq_constraints)

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

function jacobian_structure(m::IpoptOptimizer)
    num_nlp_constraints = length(m.nlp_data.constraint_bounds)
    if num_nlp_constraints > 0
        nlp_jacobian_sparsity = MOI.jacobian_structure(m.nlp_data.evaluator)
    else
        nlp_jacobian_sparsity = []
    end

    jacobian_sparsity = Tuple{Int64,Int64}[]
    row = 1
    @append_to_jacobian_sparsity m.linear_le_constraints
    @append_to_jacobian_sparsity m.linear_ge_constraints
    @append_to_jacobian_sparsity m.linear_eq_constraints
    @append_to_jacobian_sparsity m.quadratic_le_constraints
    @append_to_jacobian_sparsity m.quadratic_ge_constraints
    @append_to_jacobian_sparsity m.quadratic_eq_constraints
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

function hessian_lagrangian_structure(m::IpoptOptimizer)
    hessian_sparsity = Tuple{Int64,Int64}[]
    if m.objective !== nothing
        append_to_hessian_sparsity!(hessian_sparsity, m.objective)
    end
    for (quad, set) in m.quadratic_le_constraints
        append_to_hessian_sparsity!(hessian_sparsity, quad)
    end
    for (quad, set) in m.quadratic_ge_constraints
        append_to_hessian_sparsity!(hessian_sparsity, quad)
    end
    for (quad, set) in m.quadratic_eq_constraints
        append_to_hessian_sparsity!(hessian_sparsity, quad)
    end
    nlp_hessian_sparsity = MOI.hessian_lagrangian_structure(m.nlp_data.evaluator)
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

function eval_objective(m::IpoptOptimizer, x)
    @assert !(m.nlp_data.has_objective && m.objective !== nothing)
    if m.nlp_data.has_objective
        return MOI.eval_objective(m.nlp_data.evaluator, x)
    elseif m.objective !== nothing
        return eval_function(m.objective, x)
    else
        error("No objective function set!")
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

function eval_objective_gradient(m::IpoptOptimizer, grad, x)
    @assert !(m.nlp_data.has_objective && m.objective !== nothing)
    if m.nlp_data.has_objective
        MOI.eval_objective_gradient(m.nlp_data.evaluator, grad, x)
    elseif m.objective !== nothing
        fill_gradient!(grad, x, m.objective)
    else
        error("No objective function set!")
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

function eval_constraint(m::IpoptOptimizer, g, x)
    row = 1
    @eval_function m.linear_le_constraints
    @eval_function m.linear_ge_constraints
    @eval_function m.linear_eq_constraints
    @eval_function m.quadratic_le_constraints
    @eval_function m.quadratic_ge_constraints
    @eval_function m.quadratic_eq_constraints
    nlp_g = view(g, row:length(g))
    MOI.eval_constraint(m.nlp_data.evaluator, nlp_g, x)
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
            $esc_offset += fill_constraint_jacobian!($(esc(:values)), $esc_offset, $(esc(:x)), func)
        end
    end
end

function eval_constraint_jacobian(m::IpoptOptimizer, values, x)
    offset = 0
    @fill_constraint_jacobian m.linear_le_constraints
    @fill_constraint_jacobian m.linear_ge_constraints
    @fill_constraint_jacobian m.linear_eq_constraints
    @fill_constraint_jacobian m.quadratic_le_constraints
    @fill_constraint_jacobian m.quadratic_ge_constraints
    @fill_constraint_jacobian m.quadratic_eq_constraints

    nlp_values = view(values, 1+offset:length(values))
    MOI.eval_constraint_jacobian(m.nlp_data.evaluator, nlp_values, x)
    return
end

function fill_hessian_lagrangian!(values, start_offset, scale_factor, ::Union{MOI.SingleVariable,MOI.ScalarAffineFunction,Nothing})
    return 0
end

function fill_hessian_lagrangian!(values, start_offset, scale_factor, quad::MOI.ScalarQuadraticFunction)
    for i in 1:length(quad.quadratic_terms)
        values[start_offset + i] = scale_factor*quad.quadratic_terms[i].coefficient
    end
    return length(quad.quadratic_terms)
end

function eval_hessian_lagrangian(m::IpoptOptimizer, values, x, obj_factor, lambda)
    offset = fill_hessian_lagrangian!(values, 0, obj_factor, m.objective)
    for (i, (quad, set)) in enumerate(m.quadratic_le_constraints)
        offset += fill_hessian_lagrangian!(values, offset, lambda[i+quadratic_le_offset(m)], quad)
    end
    for (i, (quad, set)) in enumerate(m.quadratic_ge_constraints)
        offset += fill_hessian_lagrangian!(values, offset, lambda[i+quadratic_ge_offset(m)], quad)
    end
    for (i, (quad, set)) in enumerate(m.quadratic_eq_constraints)
        offset += fill_hessian_lagrangian!(values, offset, lambda[i+quadratic_eq_offset(m)], quad)
    end
    nlp_values = view(values, 1 + offset : length(values))
    nlp_lambda = view(lambda, 1 + nlp_constraint_offset(m) : length(lambda))
    MOI.eval_hessian_lagrangian(m.nlp_data.evaluator, nlp_values, x, obj_factor, nlp_lambda)
end

function constraint_bounds(m::IpoptOptimizer)
    constraint_lb = Float64[]
    constraint_ub = Float64[]
    for (func, set) in m.linear_le_constraints
        push!(constraint_lb, -Inf)
        push!(constraint_ub, set.upper)
    end
    for (func, set) in m.linear_ge_constraints
        push!(constraint_lb, set.lower)
        push!(constraint_ub, Inf)
    end
    for (func, set) in m.linear_eq_constraints
        push!(constraint_lb, set.value)
        push!(constraint_ub, set.value)
    end
    for (func, set) in m.quadratic_le_constraints
        push!(constraint_lb, -Inf)
        push!(constraint_ub, set.upper)
    end
    for (func, set) in m.quadratic_ge_constraints
        push!(constraint_lb, set.lower)
        push!(constraint_ub, Inf)
    end
    for (func, set) in m.quadratic_eq_constraints
        push!(constraint_lb, set.value)
        push!(constraint_ub, set.value)
    end
    for bound in m.nlp_data.constraint_bounds
        push!(constraint_lb, bound.lower)
        push!(constraint_ub, bound.upper)
    end
    return constraint_lb, constraint_ub
end

function MOI.optimize!(m::IpoptOptimizer)
    # TODO: Reuse m.inner for incremental solves if possible.
    num_variables = length(m.variable_info)
    num_linear_le_constraints = length(m.linear_le_constraints)
    num_linear_ge_constraints = length(m.linear_ge_constraints)
    num_linear_eq_constraints = length(m.linear_eq_constraints)
    nlp_row_offset = nlp_constraint_offset(m)
    num_quadratic_constraints = nlp_constraint_offset(m) - quadratic_le_offset(m)
    num_nlp_constraints = length(m.nlp_data.constraint_bounds)
    num_constraints = num_nlp_constraints + nlp_row_offset

    evaluator = m.nlp_data.evaluator
    features = MOI.features_available(evaluator)
    has_hessian = (:Hess in features)
    init_feat = [:Grad]
    has_hessian && push!(init_feat, :Hess)
    num_nlp_constraints > 0 && push!(init_feat, :Jac)

    MOI.initialize!(evaluator, init_feat)
    jacobian_sparsity = jacobian_structure(m)
    hessian_sparsity = has_hessian ? hessian_lagrangian_structure(m) : []

    # Objective callback
    if m.sense == MOI.MinSense
        objective_scale = 1.0
    elseif m.sense == MOI.MaxSense
        objective_scale = -1.0
    else
        error("FeasibilitySense not yet supported")
    end

    eval_f_cb(x) = objective_scale*eval_objective(m, x)

    # Objective gradient callback
    function eval_grad_f_cb(x, grad_f)
        eval_objective_gradient(m, grad_f, x)
        scale!(grad_f,objective_scale)
    end

    # Constraint value callback
    eval_g_cb(x, g) = eval_constraint(m, g, x)

    # Jacobian callback
    function eval_jac_g_cb(x, mode, rows, cols, values)
        if mode == :Structure
            for i in 1:length(jacobian_sparsity)
                rows[i] = jacobian_sparsity[i][1]
                cols[i] = jacobian_sparsity[i][2]
            end
        else
            eval_constraint_jacobian(m, values, x)
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
                eval_hessian_lagrangian(m, values, x, objective_scale*obj_factor, lambda)
            end
        end
    else
        eval_h_cb = nothing
    end

    x_l = [v.lower_bound for v in m.variable_info]
    x_u = [v.upper_bound for v in m.variable_info]

    constraint_lb, constraint_ub = constraint_bounds(m)

    m.inner = createProblem(num_variables, x_l, x_u, num_constraints,
                            constraint_lb, constraint_ub,
                            length(jacobian_sparsity),
                            length(hessian_sparsity),
                            eval_f_cb, eval_g_cb, eval_grad_f_cb, eval_jac_g_cb,
                            eval_h_cb)
    if !has_hessian
        addOption(m.inner, "hessian_approximation", "limited-memory")
    end
    if num_nlp_constraints == 0 && num_quadratic_constraints == 0
        addOption(m.inner, "jac_c_constant", "yes")
        addOption(m.inner, "jac_d_constant", "yes")
        if !m.nlp_data.has_objective
            # We turn on this option if all constraints are linear and the
            # objective is linear or quadratic. From the documentation, it's
            # unclear if it may also apply if the constraints are at most
            # quadratic.
            addOption(m.inner, "hessian_constant", "yes")
        end
    end

    m.inner.x = [v.start for v in m.variable_info]

    for (name,value) in m.options
        sname = string(name)
        if match(r"(^resto_)", sname) != nothing
            sname = replace(sname, r"(^resto_)", "resto.")
        end
        addOption(m.inner, sname, value)
    end
    solveProblem(m.inner)
end

MOI.canget(m::IpoptOptimizer, ::MOI.TerminationStatus) = m.inner !== nothing

function MOI.get(m::IpoptOptimizer, ::MOI.TerminationStatus)
    status = ApplicationReturnStatus[m.inner.status]
    if status in (:Solve_Succeeded,
                  :Feasible_Point_Found,
                  :Infeasible_Problem_Detected) # A result is available.
        return MOI.Success
    elseif status == :Solved_To_Acceptable_Level
        return MOI.AlmostSuccess
    elseif status == :Search_Direction_Becomes_Too_Small
        return MOI.NumericalError
    elseif status == :Diverging_Iterates
        return MOI.NormLimit
    elseif status == :User_Requested_Stop
        return MOI.Interrupted
    elseif status == :Maximum_Iterations_Exceeded
        return MOI.IterationLimit
    elseif status == :Maximum_CpuTime_Exceeded
        return MOI.TimeLimit
    elseif status == :Restoration_Failed
        return MOI.NumericalError
    elseif status == :Error_In_Step_Computation
        return MOI.NumericalError
    elseif status == :Invalid_Option
        return MOI.InvaidOption
    elseif status == :Not_Enough_Degrees_Of_Freedom
        return MOI.InvalidModel
    elseif status == :Invalid_Problem_Definition
        return MOI.InvalidModel
    elseif status == :Unrecoverable_Exception
        return MOI.OtherError
    elseif status == :NonIpopt_Exception_Thrown
        return MOI.OtherError
    elseif status == :Insufficient_Memory
        return MOI.MemoryLimit
    else
        error("Unrecognized Ipopt status $status")
    end
end

MOI.canget(m::IpoptOptimizer, ::MOI.ResultCount) = m.inner !== nothing
# Ipopt always has an iterate available.
MOI.get(m::IpoptOptimizer, ::MOI.ResultCount) = 1

MOI.canget(m::IpoptOptimizer, ::MOI.PrimalStatus) = m.inner !== nothing
function MOI.get(m::IpoptOptimizer, ::MOI.PrimalStatus)
    status = ApplicationReturnStatus[m.inner.status]
    if status == :Solve_Succeeded
        return MOI.FeasiblePoint
    elseif status == :Feasible_Point_Found
        return MOI.FeasiblePoint
    elseif status == :Solved_To_Acceptable_Level
        # Solutions are only guaranteed to satisfy the "acceptable" convergence
        # tolerances.
        return MOI.NearlyFeasiblePoint
    elseif status == :Infeasible_Problem_Detected
        return MOI.InfeasiblePoint
    else
        return MOI.Unknown
    end
end

MOI.canget(m::IpoptOptimizer, ::MOI.DualStatus) = m.inner !== nothing
function MOI.get(m::IpoptOptimizer, ::MOI.DualStatus)
    status = ApplicationReturnStatus[m.inner.status]
    if status == :Solve_Succeeded
        return MOI.FeasiblePoint
    elseif status == :Feasible_Point_Found
        return MOI.FeasiblePoint
    elseif status == :Solved_To_Acceptable_Level
        # Solutions are only guaranteed to satisfy the "acceptable" convergence
        # tolerances.
        return MOI.NearlyFeasiblePoint
    elseif status == :Infeasible_Problem_Detected
        # TODO: What is the interpretation of the dual in this case?
        return MOI.Unknown
    else
        return MOI.Unknown
    end
end

MOI.canget(m::IpoptOptimizer, ::MOI.ObjectiveValue) = m.inner !== nothing
function MOI.get(m::IpoptOptimizer, ::MOI.ObjectiveValue)
    scale = (m.sense == MOI.MaxSense) ? -1 : 1
    return scale*m.inner.obj_val
end

# TODO: This is a bit off, because the variable primal should be available
# only after a solve. If m.inner is initialized but we haven't solved, then
# the primal values we return do not have the intended meaning.
function MOI.canget(m::IpoptOptimizer, ::MOI.VariablePrimal, ::Type{MOI.VariableIndex})
    return m.inner !== nothing
end

function MOI.get(m::IpoptOptimizer, ::MOI.VariablePrimal, vi::MOI.VariableIndex)
    check_inbounds(m, vi)
    return m.inner.x[vi.value]
end

macro define_constraint_primal(function_type, set_type, prefix)
    constraint_array = Symbol(string(prefix) * "_constraints")
    offset_function = Symbol(string(prefix) * "_offset")
    quote
        function MOI.canget(m::IpoptOptimizer, ::MOI.ConstraintPrimal, ::Type{MOI.ConstraintIndex{$function_type, $set_type}})
            return m.inner != nothing
        end
        function MOI.get(m::IpoptOptimizer, ::MOI.ConstraintPrimal, ci::MOI.ConstraintIndex{$function_type, $set_type})
            if !(1 <= ci.value <= length(m.$(constraint_array)))
                error("Invalid constraint index ", ci.value)
            end
            return m.inner.g[ci.value + $offset_function(m)]
        end
    end
end

@define_constraint_primal MOI.ScalarAffineFunction{Float64} MOI.LessThan{Float64} linear_le
@define_constraint_primal MOI.ScalarAffineFunction{Float64} MOI.GreaterThan{Float64} linear_ge
@define_constraint_primal MOI.ScalarAffineFunction{Float64} MOI.EqualTo{Float64} linear_eq
@define_constraint_primal MOI.ScalarQuadraticFunction{Float64} MOI.LessThan{Float64} quadratic_le
@define_constraint_primal MOI.ScalarQuadraticFunction{Float64} MOI.GreaterThan{Float64} quadratic_ge
@define_constraint_primal MOI.ScalarQuadraticFunction{Float64} MOI.EqualTo{Float64} quadratic_eq

function MOI.canget(m::IpoptOptimizer, ::MOI.ConstraintPrimal,
    ::Union{Type{MOI.ConstraintIndex{MOI.SingleVariable,MOI.LessThan{Float64}}},
            Type{MOI.ConstraintIndex{MOI.SingleVariable,MOI.GreaterThan{Float64}}},
            Type{MOI.ConstraintIndex{MOI.SingleVariable,MOI.EqualTo{Float64}}}})
    return m.inner !== nothing
end

function MOI.get(m::IpoptOptimizer, ::MOI.ConstraintPrimal, ci::MOI.ConstraintIndex{MOI.SingleVariable, MOI.LessThan{Float64}})
    vi = MOI.VariableIndex(ci.value)
    check_inbounds(m, vi)
    if !has_upper_bound(m, vi)
        error("Variable $vi has no upper bound -- ConstraintPrimal not defined.")
    end
    return m.inner.x[vi.value]
end

function MOI.get(m::IpoptOptimizer, ::MOI.ConstraintPrimal, ci::MOI.ConstraintIndex{MOI.SingleVariable, MOI.GreaterThan{Float64}})
    vi = MOI.VariableIndex(ci.value)
    check_inbounds(m, vi)
    if !has_lower_bound(m, vi)
        error("Variable $vi has no lower bound -- ConstraintPrimal not defined.")
    end
    return m.inner.x[vi.value]
end

function MOI.get(m::IpoptOptimizer, ::MOI.ConstraintPrimal, ci::MOI.ConstraintIndex{MOI.SingleVariable, MOI.EqualTo{Float64}})
    vi = MOI.VariableIndex(ci.value)
    check_inbounds(m, vi)
    if !is_fixed(m, vi)
        error("Variable $vi is not fixed -- ConstraintPrimal not defined.")
    end
    return m.inner.x[vi.value]
end

function MOI.canget(m::IpoptOptimizer, ::MOI.ConstraintDual,
    ::Union{Type{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.LessThan{Float64}}},
            Type{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.GreaterThan{Float64}}},
            Type{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}}}})
    return m.inner !== nothing
end

function MOI.get(m::IpoptOptimizer, ::MOI.ConstraintDual, ci::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.LessThan{Float64}})
    @assert 1 <= ci.value <= length(m.linear_le_constraints)
    # TODO: Unable to find documentation in Ipopt about the signs of duals.
    # Rescaling by -1 here seems to pass the MOI tests.
    return -1*m.inner.mult_g[ci.value + linear_le_offset(m)]
end

function MOI.get(m::IpoptOptimizer, ::MOI.ConstraintDual, ci::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.GreaterThan{Float64}})
    @assert 1 <= ci.value <= length(m.linear_ge_constraints)
    # TODO: Unable to find documentation in Ipopt about the signs of duals.
    # Rescaling by -1 here seems to pass the MOI tests.
    return -1*m.inner.mult_g[ci.value + linear_ge_offset(m)]
end

function MOI.get(m::IpoptOptimizer, ::MOI.ConstraintDual, ci::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}})
    @assert 1 <= ci.value <= length(m.linear_eq_constraints)
    # TODO: Rescaling by -1 for consistency, but I don't know if this is covered by tests.
    return -1*m.inner.mult_g[ci.value + linear_eq_offset(m)]
end

function MOI.canget(m::IpoptOptimizer, ::MOI.ConstraintDual,
    ::Union{Type{MOI.ConstraintIndex{MOI.SingleVariable,MOI.LessThan{Float64}}},
            Type{MOI.ConstraintIndex{MOI.SingleVariable,MOI.GreaterThan{Float64}}},
            Type{MOI.ConstraintIndex{MOI.SingleVariable,MOI.EqualTo{Float64}}}})
    return m.inner !== nothing
end

function MOI.get(m::IpoptOptimizer, ::MOI.ConstraintDual, ci::MOI.ConstraintIndex{MOI.SingleVariable, MOI.LessThan{Float64}})
    vi = MOI.VariableIndex(ci.value)
    check_inbounds(m, vi)
    if !has_upper_bound(m, vi)
        error("Variable $vi has no upper bound -- ConstraintDual not defined.")
    end
    # MOI convention is for feasible LessThan duals to be nonpositive.
    return -1*m.inner.mult_x_U[vi.value]
end

function MOI.get(m::IpoptOptimizer, ::MOI.ConstraintDual, ci::MOI.ConstraintIndex{MOI.SingleVariable, MOI.GreaterThan{Float64}})
    vi = MOI.VariableIndex(ci.value)
    check_inbounds(m, vi)
    if !has_lower_bound(m, vi)
        error("Variable $vi has no lower bound -- ConstraintDual not defined.")
    end
    return m.inner.mult_x_L[vi.value]
end

function MOI.get(m::IpoptOptimizer, ::MOI.ConstraintDual, ci::MOI.ConstraintIndex{MOI.SingleVariable, MOI.EqualTo{Float64}})
    vi = MOI.VariableIndex(ci.value)
    check_inbounds(m, vi)
    if !is_fixed(m, vi)
        error("Variable $vi is not fixed -- ConstraintDual not defined.")
    end
    return m.inner.mult_x_L[vi.value] - m.inner.mult_x_U[vi.value]
end

MOI.canget(m::IpoptOptimizer, ::MOI.NLPBlockDual) = m.inner !== nothing

function MOI.get(m::IpoptOptimizer, ::MOI.NLPBlockDual)
    return -1*m.inner.mult_g[1+nlp_constraint_offset(m):end]
end
