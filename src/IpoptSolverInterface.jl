# Standard LP interface
importall MathProgBase.SolverInterface

###############################################################################
# Solver objects
export IpoptSolver
immutable IpoptSolver <: AbstractMathProgSolver
    options
end
IpoptSolver(;kwargs...) = IpoptSolver(kwargs)

type IpoptMathProgModel <: AbstractNonlinearModel
    inner::IpoptProblem
    numvar::Int
    numconstr::Int
    warmstart::Vector{Float64}
    options

    function IpoptMathProgModel(;options...)
        model = new()
        model.options   = options
        model.numvar    = 0
        model.numconstr = 0
        model.warmstart = Float64[]
        model
    end
end
NonlinearModel(s::IpoptSolver) = IpoptMathProgModel(;s.options...)
LinearQuadraticModel(s::IpoptSolver) = NonlinearToLPQPBridge(NonlinearModel(s))

###############################################################################
# Begin interface implementation

# generic nonlinear interface
function loadproblem!(m::IpoptMathProgModel, numVar::Integer, numConstr::Integer, x_l, x_u, g_lb, g_ub, sense::Symbol, d::AbstractNLPEvaluator)

    m.numvar = numVar
    features = features_available(d)
    has_hessian = (:Hess in features)
    init_feat = [:Grad]
    has_hessian && push!(init_feat, :Hess)
    numConstr > 0 && push!(init_feat, :Jac)

    initialize(d, init_feat)
    Ihess, Jhess = has_hessian ? hesslag_structure(d) : (Int[], Int[])
    Ijac, Jjac = numConstr > 0 ? jac_structure(d) : (Int[], Int[])
    @assert length(Ijac) == length(Jjac)
    @assert length(Ihess) == length(Jhess)
    @assert sense == :Min || sense == :Max

    # Objective callback
    scl = (sense == :Min) ? 1.0 : -1.0
    eval_f_cb(x) = scl*eval_f(d,x)

    # Objective gradient callback
    eval_grad_f_cb(x, grad_f) = (eval_grad_f(d, grad_f, x); scale!(grad_f,scl))

    # Constraint value callback
    eval_g_cb(x, g) = eval_g(d, g, x)

    # Jacobian callback
    function eval_jac_g_cb(x, mode, rows, cols, values)
        if mode == :Structure
            for i in 1:length(Ijac)
                rows[i] = Ijac[i]
                cols[i] = Jjac[i]
            end
        else
            eval_jac_g(d, values, x)
        end
    end

    if has_hessian
        # Hessian callback
        function eval_h_cb(x, mode, rows, cols, obj_factor,
            lambda, values)
            if mode == :Structure
                for i in 1:length(Ihess)
                    rows[i] = Ihess[i]
                    cols[i] = Jhess[i]
                end
            else
                obj_factor *= scl
                eval_hesslag(d, values, x, obj_factor, lambda)
            end
        end
    else
        eval_h_cb = nothing
    end


    m.inner = createProblem(numVar, float(x_l), float(x_u), numConstr,
    float(g_lb), float(g_ub), length(Ijac), length(Ihess),
    eval_f_cb, eval_g_cb, eval_grad_f_cb, eval_jac_g_cb,
    eval_h_cb)
    m.inner.sense = sense
    if !has_hessian
        addOption(m.inner, "hessian_approximation", "limited-memory")
    end

end

getsense(m::IpoptMathProgModel) = m.inner.sense
numvar(m::IpoptMathProgModel) = m.numvar
numconstr(m::IpoptMathProgModel) = m.numconstr

numlinconstr(m::IpoptMathProgModel) = 0

numquadconstr(m::IpoptMathProgModel) = 0

function optimize!(m::IpoptMathProgModel)
    copy!(m.inner.x, m.warmstart) # set warmstart
    for (name,value) in m.options
        sname = string(name)
        if match(r"(^resto_)", sname) != nothing
            sname = replace(sname, r"(^resto_)", "resto.")
        end
        addOption(m.inner, sname, value)
    end
    solveProblem(m.inner)
end

function status(m::IpoptMathProgModel)
    # Map all the possible return codes, as enumerated in
    # Ipopt.ApplicationReturnStatus, to the MPB statuses:
    # :Optimal, :Infeasible, :Unbounded, :UserLimit, and :Error
    stat_sym = ApplicationReturnStatus[m.inner.status]
    if  stat_sym == :Solve_Succeeded ||
        stat_sym == :Solved_To_Acceptable_Level
        return :Optimal
    elseif stat_sym == :Infeasible_Problem_Detected
        return :Infeasible
    elseif stat_sym == :Diverging_Iterates
        return :Unbounded
        # Things that are more likely to be fixable by changing
        # a parameter will be treated as UserLimit, although
        # some are error-like too.
    elseif stat_sym == :User_Requested_Stop ||
        stat_sym == :Maximum_Iterations_Exceeded ||
        stat_sym == :Maximum_CpuTime_Exceeded
        return :UserLimit
    else
        # Default is to not mislead user that it worked
        # Includes:
        #   :Search_Direction_Becomes_Too_Small
        #   :Feasible_Point_Found
        #   :Restoration_Failed
        #   :Error_In_Step_Computation
        #   :Not_Enough_Degrees_Of_Freedom
        #   :Invalid_Problem_Definition
        #   :Invalid_Option
        #   :Invalid_Number_Detected
        #   :Unrecoverable_Exception
        #   :NonIpopt_Exception_Thrown
        #   :Insufficient_Memory
        #   :Internal_Error
        warn("Ipopt finished with status $stat_sym")
        return :Error
    end

end
getobjval(m::IpoptMathProgModel) = m.inner.obj_val * (m.inner.sense == :Max ? -1 : +1)
getsolution(m::IpoptMathProgModel) = m.inner.x

function getreducedcosts(m::IpoptMathProgModel)
    sense = m.inner.sense
    redcost = m.inner.mult_x_U - m.inner.mult_x_L
    return sense == :Max ? redcost : -redcost
end
function getconstrduals(m::IpoptMathProgModel)
    v = m.inner.mult_g # return multipliers for all constraints
    return m.inner.sense == :Max ? copy(v) : -v
end

getrawsolver(m::IpoptMathProgModel) = m.inner
setwarmstart!(m::IpoptMathProgModel, x) = (m.warmstart = x)
