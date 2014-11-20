# Standard LP interface
importall MathProgBase.SolverInterface

###############################################################################
# Solver objects
export IpoptSolver
immutable IpoptSolver <: AbstractMathProgSolver
    options
end
IpoptSolver(;kwargs...) = IpoptSolver(kwargs)

type IpoptMathProgModel <: AbstractMathProgModel
    inner::Any
    LPdata
    Qobj::(Vector{Int},Vector{Int},Vector{Float64})
    state::Symbol # Uninitialized, LoadLinear, LoadNonlinear
    numvar::Int
    numconstr::Int
    warmstart::Vector{Float64}
    options
end
function IpoptMathProgModel(;options...)
    return IpoptMathProgModel(nothing,nothing,(Int[],Int[],Float64[]),:Uninitialized,0,0,Float64[],options)
end
model(s::IpoptSolver) = IpoptMathProgModel(;s.options...)
export model

###############################################################################
# Begin interface implementation
function loadproblem!(model::IpoptMathProgModel, A, l, u, c, lb, ub, sense)
    model.LPdata = (A,l,u,c,lb,ub,sense)
    model.state = :LoadLinear
    model.numvar = size(A,2)
    model.numconstr = size(A,1)
end

function setquadobj!(model::IpoptMathProgModel, rowidx, colidx, quadval)
    @assert model.state == :LoadLinear # must be called after loadproblem!
    model.Qobj = (rowidx, colidx, quadval)
end

function createQPcallbacks(model::IpoptMathProgModel)
    @assert model.state == :LoadLinear
    A,l,u,c,lb,ub,sense = model.LPdata
    Asparse = convert(SparseMatrixCSC{Float64,Int32}, A)::SparseMatrixCSC{Float64,Int32}
    n = int(Asparse.n)
    m = int(Asparse.m)
    nnz = int(length(Asparse.rowval))
    c_correct = float(c)::Vector{Float64}
    Qi,Qj,Qv = model.Qobj
    if sense == :Max
        c_correct .*= -1.0
        Qv .*= -1.0
    end
    @assert length(Qi) == length(Qj) == length(Qv)



    # Objective callback
    function eval_f(x)
        obj = dot(x,c_correct)
        for k in 1:length(Qi)
            if Qi[k] == Qj[k]
                obj += Qv[k]*x[Qi[k]]*x[Qi[k]]/2
            else
                obj += Qv[k]*x[Qi[k]]*x[Qj[k]]
            end
        end
        return obj
    end

    # Objective gradient callback
    function eval_grad_f(x, grad_f)
        for j = 1:n
            grad_f[j] = c_correct[j]
        end
        for k in 1:length(Qi)
            if Qi[k] == Qj[k]
                grad_f[Qi[k]] += Qv[k]*x[Qj[k]]
            else
                grad_f[Qi[k]] += Qv[k]*x[Qj[k]]
                grad_f[Qj[k]] += Qv[k]*x[Qi[k]]
            end
        end
    end

    # Constraint value callback
    function eval_g(x, g)
        g_val = A*x
        for i = 1:m
            g[i] = g_val[i]
        end
    end

    # Jacobian callback
    function eval_jac_g(x, mode, rows, cols, values)
        if mode == :Structure
            # Convert column wise sparse to triple format
            idx = 1
            for col = 1:n
                for pos = Asparse.colptr[col]:(Asparse.colptr[col+1]-1)
                    rows[idx] = Asparse.rowval[pos]
                    cols[idx] = col
                    idx += 1
                end
            end
        else
            # Values
            idx = 1
            for col = 1:n
                for pos = Asparse.colptr[col]:(Asparse.colptr[col+1]-1)
                    values[idx] = Asparse.nzval[pos]
                    idx += 1
                end
            end
        end
    end

    # Hessian callback
    function eval_h(x, mode, rows, cols, obj_factor,
        lambda, values)
        if mode == :Structure
            for k in 1:length(Qi)
                rows[k] = Qi[k]
                cols[k] = Qj[k]
            end
        else
            for k in 1:length(Qi)
                values[k] = obj_factor*Qv[k]
            end
        end
    end

    x_L = float(l)
    x_U = float(u)
    g_L = float(lb)
    g_U = float(ub)
    model.inner = createProblem(n, x_L, x_U, m, g_L, g_U, nnz, length(Qv),
    eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h)
    model.inner.sense = sense
    addOption(model.inner, "jac_c_constant", "yes") # all equality constraints linear
    addOption(model.inner, "jac_d_constant", "yes") # all inequality constraints linear
    addOption(model.inner, "hessian_constant", "yes")
    addOption(model.inner, "mehrotra_algorithm", "yes")

end

# generic nonlinear interface
function loadnonlinearproblem!(m::IpoptMathProgModel, numVar::Integer, numConstr::Integer, x_l, x_u, g_lb, g_ub, sense::Symbol, d::AbstractNLPEvaluator)

    initialize(d, [:Grad, :Jac, :Hess])
    Ijac, Jjac = jac_structure(d)
    Ihess, Jhess = hesslag_structure(d)
    @assert length(Ijac) == length(Jjac)
    @assert length(Ihess) == length(Jhess)
    @assert sense == :Min || sense == :Max

    # Objective callback
    if sense == :Min
        eval_f_cb(x) = eval_f(d,x)
    else
        eval_f_cb(x) = -eval_f(d,x)
    end

    # Objective gradient callback
    if sense == :Min
        eval_grad_f_cb(x, grad_f) = eval_grad_f(d, grad_f, x)
    else
        eval_grad_f_cb(x, grad_f) = (eval_grad_f(d, grad_f, x); scale!(grad_f,-1))
    end


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

    # Hessian callback
    function eval_h_cb(x, mode, rows, cols, obj_factor,
        lambda, values)
        if mode == :Structure
            for i in 1:length(Ihess)
                rows[i] = Ihess[i]
                cols[i] = Jhess[i]
            end
        else
            if sense == :Max
                obj_factor *= -1
            end
            eval_hesslag(d, values, x, obj_factor, lambda)
        end
    end


    m.inner = createProblem(numVar, float(x_l), float(x_u), numConstr,
    float(g_lb), float(g_ub), length(Ijac), length(Ihess),
    eval_f_cb, eval_g_cb, eval_grad_f_cb, eval_jac_g_cb,
    eval_h_cb)
    m.inner.sense = sense
    
    m.state = :LoadNonlinear

end

getsense(m::IpoptMathProgModel) = m.inner.sense
numvar(m::IpoptMathProgModel) = m.numvar
numconstr(m::IpoptMathProgModel) = m.numconstr

function optimize!(m::IpoptMathProgModel)
    if m.state == :LoadLinear
        createQPcallbacks(m)
    else
        @assert m.state == :LoadNonlinear
    end
    copy!(m.inner.x, m.warmstart) # set warmstart
    for (name,value) in m.options
        addOption(m.inner, string(name), value)
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
getconstrsolution(m::IpoptMathProgModel) = m.inner.g
getreducedcosts(m::IpoptMathProgModel) = m.inner.mult_x_U - m.inner.mult_x_L
# TODO: only valid when all constraints linear
getconstrduals(m::IpoptMathProgModel) = m.inner.mult_g
getrawsolver(m::IpoptMathProgModel) = m.inner
setwarmstart!(m::IpoptMathProgModel, x) = (m.warmstart = x)
