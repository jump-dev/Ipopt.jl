using MathOptInterface
const MOI = MathOptInterface

mutable struct VariableInfo
    lower_bound::Float64
    upper_bound::Float64
    start::Float64
end
VariableInfo() = VariableInfo(-Inf, Inf, NaN)

export IpoptOptimizer
mutable struct IpoptOptimizer <: MOI.AbstractOptimizer
    inner::Union{IpoptProblem,Nothing}
    variable_info::Vector{VariableInfo}
    nlp_data::Union{MOI.NLPBlockData,Nothing}
    sense::MOI.OptimizationSense
    options
end

IpoptOptimizer(;options...) = IpoptOptimizer(nothing, [], nothing, MOI.FeasibilitySense, options)

MOI.supports(::IpoptOptimizer, ::MOI.NLPBlock) = true
MOI.supportsconstraint(::IpoptOptimizer, ::Type{MOI.SingleVariable}, ::Type{MOI.LessThan{Float64}}) = true
MOI.supportsconstraint(::IpoptOptimizer, ::Type{MOI.SingleVariable}, ::Type{MOI.GreaterThan{Float64}}) = true
# TODO: The distinction between supportsconstraint and canaddconstraint is maybe too subtle.
MOI.canaddconstraint(::IpoptOptimizer, ::Type{MOI.SingleVariable}, ::Type{MOI.LessThan{Float64}}) = true
MOI.canaddconstraint(::IpoptOptimizer, ::Type{MOI.SingleVariable}, ::Type{MOI.GreaterThan{Float64}}) = true

MOI.canset(::IpoptOptimizer, ::MOI.VariablePrimalStart, ::Type{MOI.VariableIndex}) = true

MOI.canget(::IpoptOptimizer, ::MOI.NumberOfVariables) = true
MOI.get(m::IpoptOptimizer, ::MOI.NumberOfVariables) = length(m.variable_info)

function MOI.set!(m::IpoptOptimizer, ::MOI.ObjectiveSense, sense::MOI.OptimizationSense)
    m.sense = sense
    return
end

function MOI.empty!(m::IpoptOptimizer)
    m.inner = nothing
    m.variable_info = []
    m.nlp_data = nothing
    m.sense = MOI.FeasibilitySense
end

function MOI.isempty(m::IpoptOptimizer)
    return isempty(m.variable_info) && m.nlp_data === nothing && m.sense == MOI.FeasibilitySense
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

function MOI.addconstraint!(m::IpoptOptimizer, v::MOI.SingleVariable, lt::MOI.LessThan{Float64})
    vi = v.variable
    check_inbounds(m, vi)
    if m.variable_info[vi.value].upper_bound != Inf
        error("Upper bound on variable $vi already exists.")
    end
    m.variable_info[vi.value].upper_bound = lt.upper
    return MOI.ConstraintIndex{MOI.SingleVariable, MOI.LessThan{Float64}}(vi.value)
end

function MOI.addconstraint!(m::IpoptOptimizer, v::MOI.SingleVariable, gt::MOI.GreaterThan{Float64})
    vi = v.variable
    check_inbounds(m, vi)
    if m.variable_info[vi.value].lower_bound != -Inf
        error("Lower bound on variable $vi already exists.")
    end
    m.variable_info[vi.value].lower_bound = gt.lower
    return MOI.ConstraintIndex{MOI.SingleVariable, MOI.GreaterThan{Float64}}(vi.value)
end

function MOI.set!(m::IpoptOptimizer, ::MOI.VariablePrimalStart, vi::MOI.VariableIndex, value::Real)
    check_inbounds(m, vi)
    m.variable_info[vi.value].start = value
end

function MOI.set!(m::IpoptOptimizer, ::MOI.NLPBlock, nlp_data::MOI.NLPBlockData)
    m.nlp_data = nlp_data
end

function MOI.optimize!(m::IpoptOptimizer)
    # TODO: Reuse m.inner for incremental solves if possible.
    @assert m.nlp_data != nothing
    @assert m.nlp_data.has_objective

    num_variables = length(m.variable_info)
    num_constraints = length(m.nlp_data.constraint_bounds)

    evaluator = m.nlp_data.evaluator
    features = MOI.features_available(evaluator)
    has_hessian = (:Hess in features)
    init_feat = [:Grad]
    has_hessian && push!(init_feat, :Hess)
    num_constraints > 0 && push!(init_feat, :Jac)

    MOI.initialize!(evaluator, init_feat)
    hessian_sparsity = has_hessian ? MOI.hessian_lagrangian_structure(evaluator) : []
    jacobian_sparsity = num_constraints > 0 ? MOI.jacobian_structure(evaluator) : []

    # Objective callback
    if m.sense == MOI.MinSense
        objective_scale = 1.0
    elseif m.sense == MOI.MaxSense
        objective_scale = -1.0
    else
        error("FeasibilitySense not yet supported")
    end

    eval_f_cb(x) = objective_scale*MOI.eval_objective(evaluator, x)

    # Objective gradient callback
    function eval_grad_f_cb(x, grad_f)
        MOI.eval_objective_gradient(evaluator, grad_f, x)
        scale!(grad_f,objective_scale)
    end

    # Constraint value callback
    eval_g_cb(x, g) = MOI.eval_constraint(evaluator, g, x)

    # Jacobian callback
    function eval_jac_g_cb(x, mode, rows, cols, values)
        if mode == :Structure
            for i in 1:length(jacobian_sparsity)
                rows[i] = jacobian_sparsity[i][1]
                cols[i] = jacobian_sparsity[i][2]
            end
        else
            MOI.eval_constraint_jacobian(evaluator, values, x)
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
                MOI.eval_hessian_lagrangian(evaluator, values, x, obj_factor, lambda)
            end
        end
    else
        eval_h_cb = nothing
    end

    x_l = [v.lower_bound for v in m.variable_info]
    x_u = [v.upper_bound for v in m.variable_info]

    constraint_lb = [bound.lower for bound in m.nlp_data.constraint_bounds]
    constraint_ub = [bound.upper for bound in m.nlp_data.constraint_bounds]

    m.inner = createProblem(num_variables, x_l, x_u, num_constraints,
                            constraint_lb, constraint_ub,
                            length(jacobian_sparsity), length(hessian_sparsity),
                            eval_f_cb, eval_g_cb, eval_grad_f_cb, eval_jac_g_cb,
                            eval_h_cb)
    if !has_hessian
        addOption(m.inner, "hessian_approximation", "limited-memory")
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

MOI.canget(m::IpoptOptimizer, ::MOI.ObjectiveValue) = m.inner !== nothing
function MOI.get(m::IpoptOptimizer, ::MOI.ObjectiveValue)
    scale = (m.sense == MOI.MaxSense) ? -1 : 1
    return scale*m.inner.obj_val
end

function MOI.canget(m::IpoptOptimizer, ::MOI.VariablePrimal, ::Type{MOI.VariableIndex})
    return m.inner !== nothing
end

function MOI.get(m::IpoptOptimizer, ::MOI.VariablePrimal, vi::MOI.VariableIndex)
    return m.inner.x[vi.value]
end
