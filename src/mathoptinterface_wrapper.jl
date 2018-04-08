using MathOptInterface
const MOI = MathOptInterface

mutable struct VariableInfo
    lower_bound::Float64
    upper_bound::Float64
    start::Float64
end
# The default start value is zero.
VariableInfo() = VariableInfo(-Inf, Inf, 0.0)

export IpoptOptimizer
mutable struct IpoptOptimizer <: MOI.AbstractOptimizer
    inner::Union{IpoptProblem,Nothing}
    variable_info::Vector{VariableInfo}
    nlp_data::MOI.NLPBlockData
    sense::MOI.OptimizationSense
    objective::Union{MOI.ScalarAffineFunction,Nothing}
    linear_le_constraints::Vector{Tuple{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}}
    linear_ge_constraints::Vector{Tuple{MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}}}
    linear_eq_constraints::Vector{Tuple{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}}
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


IpoptOptimizer(;options...) = IpoptOptimizer(nothing, [], empty_nlp_data(), MOI.FeasibilitySense, nothing, [], [], [], options)

MOI.supports(::IpoptOptimizer, ::MOI.NLPBlock) = true
MOI.supports(::IpoptOptimizer, ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}) = true
MOI.supports(::IpoptOptimizer, ::MOI.ObjectiveSense) = true
MOI.supportsconstraint(::IpoptOptimizer, ::Type{MOI.SingleVariable}, ::Type{MOI.LessThan{Float64}}) = true
MOI.supportsconstraint(::IpoptOptimizer, ::Type{MOI.SingleVariable}, ::Type{MOI.GreaterThan{Float64}}) = true
MOI.supportsconstraint(::IpoptOptimizer, ::Type{MOI.ScalarAffineFunction{Float64}}, ::Type{MOI.LessThan{Float64}}) = true
MOI.supportsconstraint(::IpoptOptimizer, ::Type{MOI.ScalarAffineFunction{Float64}}, ::Type{MOI.GreaterThan{Float64}}) = true
MOI.supportsconstraint(::IpoptOptimizer, ::Type{MOI.ScalarAffineFunction{Float64}}, ::Type{MOI.EqualTo{Float64}}) = true

MOI.canaddvariable(::IpoptOptimizer) = true
# TODO: The distinction between supportsconstraint and canaddconstraint is maybe too subtle.
MOI.canaddconstraint(::IpoptOptimizer, ::Type{MOI.SingleVariable}, ::Type{MOI.LessThan{Float64}}) = true
MOI.canaddconstraint(::IpoptOptimizer, ::Type{MOI.SingleVariable}, ::Type{MOI.GreaterThan{Float64}}) = true
MOI.canaddconstraint(::IpoptOptimizer, ::Type{MOI.ScalarAffineFunction{Float64}}, ::Type{MOI.LessThan{Float64}}) = true
MOI.canaddconstraint(::IpoptOptimizer, ::Type{MOI.ScalarAffineFunction{Float64}}, ::Type{MOI.GreaterThan{Float64}}) = true
MOI.canaddconstraint(::IpoptOptimizer, ::Type{MOI.ScalarAffineFunction{Float64}}, ::Type{MOI.EqualTo{Float64}}) = true

MOI.canset(::IpoptOptimizer, ::MOI.VariablePrimalStart, ::Type{MOI.VariableIndex}) = true
MOI.canset(::IpoptOptimizer, ::MOI.ObjectiveSense) = true
MOI.canset(::IpoptOptimizer, ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}) = true

MOI.copy!(m::IpoptOptimizer, src::MOI.ModelLike) = MOI.Utilities.defaultcopy!(m, src)

MOI.canget(::IpoptOptimizer, ::MOI.NumberOfVariables) = true
MOI.get(m::IpoptOptimizer, ::MOI.NumberOfVariables) = length(m.variable_info)


function MOI.set!(m::IpoptOptimizer, ::MOI.ObjectiveSense, sense::MOI.OptimizationSense)
    m.sense = sense
    return
end

function MOI.empty!(m::IpoptOptimizer)
    m.inner = nothing
    m.variable_info = []
    m.nlp_data = empty_nlp_data()
    m.sense = MOI.FeasibilitySense
    m.objective = nothing
    empty!(m.linear_le_constraints)
    empty!(m.linear_ge_constraints)
    empty!(m.linear_eq_constraints)
end

function MOI.isempty(m::IpoptOptimizer)
    return isempty(m.variable_info) && m.nlp_data.evaluator isa EmptyNLPEvaluator && m.sense == MOI.FeasibilitySense
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

function check_inbounds(m::IpoptOptimizer, aff::MOI.ScalarAffineFunction)
    for v in aff.variables
        check_inbounds(m, v)
    end
end

function has_upper_bound(m::IpoptOptimizer, vi::MOI.VariableIndex)
    return m.variable_info[vi.value].upper_bound != Inf
end

function has_lower_bound(m::IpoptOptimizer, vi::MOI.VariableIndex)
    return m.variable_info[vi.value].lower_bound != -Inf
end

function MOI.addconstraint!(m::IpoptOptimizer, v::MOI.SingleVariable, lt::MOI.LessThan{Float64})
    vi = v.variable
    check_inbounds(m, vi)
    if !isfinite(lt.upper)
        error("Invalid upper bound value $(lt.upper).")
    end
    if has_upper_bound(m, vi)
        error("Upper bound on variable $vi already exists.")
    end
    m.variable_info[vi.value].upper_bound = lt.upper
    return MOI.ConstraintIndex{MOI.SingleVariable, MOI.LessThan{Float64}}(vi.value)
end

function MOI.addconstraint!(m::IpoptOptimizer, v::MOI.SingleVariable, gt::MOI.GreaterThan{Float64})
    vi = v.variable
    check_inbounds(m, vi)
    if !isfinite(gt.lower)
        error("Invalid lower bound value $(gt.lower).")
    end
    if has_lower_bound(m, vi)
        error("Lower bound on variable $vi already exists.")
    end
    m.variable_info[vi.value].lower_bound = gt.lower
    return MOI.ConstraintIndex{MOI.SingleVariable, MOI.GreaterThan{Float64}}(vi.value)
end

function MOI.addconstraint!(m::IpoptOptimizer, func::MOI.ScalarAffineFunction{Float64}, lt::MOI.LessThan{Float64})
    check_inbounds(m, func)
    push!(m.linear_le_constraints, (func, lt))
    return MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}(length(m.linear_le_constraints))
end

function MOI.addconstraint!(m::IpoptOptimizer, func::MOI.ScalarAffineFunction{Float64}, gt::MOI.GreaterThan{Float64})
    check_inbounds(m, func)
    push!(m.linear_ge_constraints, (func, gt))
    return MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}}(length(m.linear_ge_constraints))
end

function MOI.addconstraint!(m::IpoptOptimizer, func::MOI.ScalarAffineFunction{Float64}, eq::MOI.EqualTo{Float64})
    check_inbounds(m, func)
    push!(m.linear_eq_constraints, (func, eq))
    return MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}(length(m.linear_eq_constraints))
end

function MOI.set!(m::IpoptOptimizer, ::MOI.VariablePrimalStart, vi::MOI.VariableIndex, value::Real)
    check_inbounds(m, vi)
    m.variable_info[vi.value].start = value
    return
end

function MOI.set!(m::IpoptOptimizer, ::MOI.NLPBlock, nlp_data::MOI.NLPBlockData)
    m.nlp_data = nlp_data
    return
end

function MOI.set!(m::IpoptOptimizer, ::MOI.ObjectiveFunction, func::MOI.ScalarAffineFunction)
    check_inbounds(m, func)
    m.objective = func
    return
end

# In setting up the data for Ipopt, we order the constraints as follows:
# - linear_le_constraints
# - linear_ge_constraints
# - linear_eq_constraints
# - nonlinear constraints from nlp_data

linear_le_offset(m::IpoptOptimizer) = 0
linear_ge_offset(m::IpoptOptimizer) = length(m.linear_le_constraints)
linear_eq_offset(m::IpoptOptimizer) = length(m.linear_le_constraints) + length(m.linear_ge_constraints)
nlp_constraint_offset(m::IpoptOptimizer) = linear_eq_offset(m) + length(m.linear_eq_constraints)

# Convenience functions used only in optimize!

function append_to_jacobian_sparsity!(jacobian_sparsity, aff::MOI.ScalarAffineFunction, row)
    for variable_index in aff.variables
        push!(jacobian_sparsity, (row, variable_index.value))
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
    for (aff, set) in m.linear_le_constraints
        append_to_jacobian_sparsity!(jacobian_sparsity, aff, row)
        row += 1
    end
    for (aff, set) in m.linear_ge_constraints
        append_to_jacobian_sparsity!(jacobian_sparsity, aff, row)
        row += 1
    end
    for (aff, set) in m.linear_eq_constraints
        append_to_jacobian_sparsity!(jacobian_sparsity, aff, row)
        row += 1
    end
    for (nlp_row, column) in nlp_jacobian_sparsity
        push!(jacobian_sparsity, (nlp_row + row - 1, column))
    end
    return jacobian_sparsity
end

function eval_function(aff::MOI.ScalarAffineFunction, x)
    function_value = aff.constant
    for i in 1:length(aff.variables)
        var_idx = aff.variables[i]
        coefficient = aff.coefficients[i]
        # Note the implicit assumtion that VariableIndex values match up with
        # x indices. This is valid because in this wrapper ListOfVariableIndices
        # is always [1, ..., NumberOfVariables].
        function_value += coefficient*x[var_idx.value]
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

function eval_objective_gradient(m::IpoptOptimizer, grad, x)
    @assert !(m.nlp_data.has_objective && m.objective !== nothing)
    if m.nlp_data.has_objective
        MOI.eval_objective_gradient(m.nlp_data.evaluator, grad, x)
    elseif m.objective !== nothing
        aff = m.objective::MOI.ScalarAffineFunction{Float64}
        fill!(grad, 0.0)
        for i in 1:length(aff.variables)
            var_idx = aff.variables[i]
            coefficient = aff.coefficients[i]
            grad[var_idx.value] += coefficient
        end
    else
        error("No objective function set!")
    end
    return
end

function eval_constraint(m::IpoptOptimizer, g, x)
    row = 1
    for (aff, set) in m.linear_le_constraints
        g[row] = eval_function(aff, x)
        row += 1
    end
    for (aff, set) in m.linear_ge_constraints
        g[row] = eval_function(aff, x)
        row += 1
    end
    for (aff, set) in m.linear_eq_constraints
        g[row] = eval_function(aff, x)
        row += 1
    end
    nlp_g = view(g, row:length(g))
    MOI.eval_constraint(m.nlp_data.evaluator, nlp_g, x)
    return
end

function eval_constraint_jacobian(m::IpoptOptimizer, values, x)
    offset = 1
    for (aff, set) in m.linear_le_constraints
        num_coefficients = length(aff.coefficients)
        values[offset:offset+num_coefficients-1] .= aff.coefficients
        offset += num_coefficients
    end
    for (aff, set) in m.linear_ge_constraints
        num_coefficients = length(aff.coefficients)
        values[offset:offset+num_coefficients-1] .= aff.coefficients
        offset += num_coefficients
    end
    for (aff, set) in m.linear_eq_constraints
        num_coefficients = length(aff.coefficients)
        values[offset:offset+num_coefficients-1] .= aff.coefficients
        offset += num_coefficients
    end
    nlp_values = view(values, offset:length(values))
    MOI.eval_constraint_jacobian(m.nlp_data.evaluator, nlp_values, x)
    return
end

function constraint_bounds(m::IpoptOptimizer)
    constraint_lb = Float64[]
    constraint_ub = Float64[]
    for (aff, set) in m.linear_le_constraints
        push!(constraint_lb, -Inf)
        push!(constraint_ub, set.upper)
    end
    for (aff, set) in m.linear_ge_constraints
        push!(constraint_lb, set.lower)
        push!(constraint_ub, Inf)
    end
    for (aff, set) in m.linear_eq_constraints
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
    nlp_hessian_sparsity = has_hessian ? MOI.hessian_lagrangian_structure(evaluator) : []

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
                for i in 1:length(nlp_hessian_sparsity)
                    rows[i] = nlp_hessian_sparsity[i][1] + nlp_row_offset
                    cols[i] = nlp_hessian_sparsity[i][2]
                end
            else
                obj_factor *= objective_scale
                nlp_lambda = view(lambda, 1 + nlp_row_offset : length(lambda))
                MOI.eval_hessian_lagrangian(evaluator, values, x, obj_factor, nlp_lambda)
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
                            length(jacobian_sparsity), length(nlp_hessian_sparsity),
                            eval_f_cb, eval_g_cb, eval_grad_f_cb, eval_jac_g_cb,
                            eval_h_cb)
    if !has_hessian
        addOption(m.inner, "hessian_approximation", "limited-memory")
    end
    if num_nlp_constraints == 0
        # TODO: this changes when we support quadratic
        addOption(m.inner, "jac_c_constant", "yes")
        addOption(m.inner, "jac_d_constant", "yes")
        addOption(m.inner, "hessian_constant", "yes")
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

function MOI.canget(m::IpoptOptimizer, ::MOI.VariablePrimal, ::Type{MOI.VariableIndex})
    return m.inner !== nothing
end

function MOI.get(m::IpoptOptimizer, ::MOI.VariablePrimal, vi::MOI.VariableIndex)
    check_inbounds(m, vi)
    return m.inner.x[vi.value]
end

function MOI.canget(m::IpoptOptimizer, ::MOI.ConstraintPrimal,
    ::Union{Type{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.LessThan{Float64}}},
            Type{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.GreaterThan{Float64}}},
            Type{MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}}}})
    return m.inner !== nothing
end

function MOI.get(m::IpoptOptimizer, ::MOI.ConstraintPrimal, ci::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.LessThan{Float64}})
    @assert 1 <= ci.value <= length(m.linear_le_constraints)
    return m.inner.g[ci.value + linear_le_offset(m)]
end

function MOI.get(m::IpoptOptimizer, ::MOI.ConstraintPrimal, ci::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.GreaterThan{Float64}})
    @assert 1 <= ci.value <= length(m.linear_ge_constraints)
    return m.inner.g[ci.value + linear_ge_offset(m)]
end

function MOI.get(m::IpoptOptimizer, ::MOI.ConstraintPrimal, ci::MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}})
    @assert 1 <= ci.value <= length(m.linear_eq_constraints)
    return m.inner.g[ci.value + linear_eq_offset(m)]
end

function MOI.canget(m::IpoptOptimizer, ::MOI.ConstraintPrimal,
    ::Union{Type{MOI.ConstraintIndex{MOI.SingleVariable,MOI.LessThan{Float64}}},
            Type{MOI.ConstraintIndex{MOI.SingleVariable,MOI.GreaterThan{Float64}}}})
    return m.inner !== nothing
end

function MOI.get(m::IpoptOptimizer, ::MOI.ConstraintPrimal, ci::MOI.ConstraintIndex{MOI.SingleVariable, MOI.LessThan{Float64}})
    vi = MOI.VariableIndex(ci.value)
    check_inbounds(m, vi)
    if !has_upper_bound(m, vi)
        error("No upper bound -- ConstraintPrimal not defined.")
    end
    return m.inner.x[vi.value]
end

function MOI.get(m::IpoptOptimizer, ::MOI.ConstraintPrimal, ci::MOI.ConstraintIndex{MOI.SingleVariable, MOI.GreaterThan{Float64}})
    vi = MOI.VariableIndex(ci.value)
    check_inbounds(m, vi)
    if !has_lower_bound(m, vi)
        error("No lower bound -- ConstraintPrimal not defined.")
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
            Type{MOI.ConstraintIndex{MOI.SingleVariable,MOI.GreaterThan{Float64}}}})
    return m.inner !== nothing
end

function MOI.get(m::IpoptOptimizer, ::MOI.ConstraintDual, ci::MOI.ConstraintIndex{MOI.SingleVariable, MOI.LessThan{Float64}})
    vi = MOI.VariableIndex(ci.value)
    check_inbounds(m, vi)
    if !has_upper_bound(m, vi)
        error("No upper bound -- ConstraintDual not defined.")
    end
    # MOI convention is for feasible LessThan duals to be nonpositive.
    return -1*m.inner.mult_x_U[vi.value]
end

function MOI.get(m::IpoptOptimizer, ::MOI.ConstraintDual, ci::MOI.ConstraintIndex{MOI.SingleVariable, MOI.GreaterThan{Float64}})
    vi = MOI.VariableIndex(ci.value)
    check_inbounds(m, vi)
    if !has_lower_bound(m, vi)
        error("No lower bound -- ConstraintDual not defined.")
    end
    return m.inner.mult_x_L[vi.value]
end
