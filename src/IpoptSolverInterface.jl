# Standard LP interface
require(joinpath(Pkg.dir("MathProgBase"),"src","MathProgSolverInterface.jl"))
importall MathProgSolverInterface

###############################################################################
# Solver objects
export IpoptSolver
immutable IpoptSolver <: AbstractMathProgSolver
  options
end
IpoptSolver(;kwargs...) = IpoptSolver(kwargs)

type IpoptMathProgModel <: AbstractMathProgModel
  inner::Any
  options
end
function IpoptMathProgModel(;options...)
  return IpoptMathProgModel(nothing,options)
end
model(s::IpoptSolver) = IpoptMathProgModel(;s.options...)
export model

###############################################################################
# Begin interface implementation
function loadproblem!(model::IpoptMathProgModel, A, l, u, c, lb, ub, sense)
  Asparse = convert(SparseMatrixCSC{Float64,Int32}, A)
  n = int(Asparse.n)
  m = int(Asparse.m)
  nnz = int(length(Asparse.rowval))
  c_correct = float(c)
  if sense == :Max
    c_correct .*= -1.0
  end


  # Objective callback
  function eval_f(x)
    return dot(x,c_correct)
  end

  # Objective gradient callback
  function eval_grad_f(x, grad_f)
    for j = 1:n
      grad_f[j] = c_correct[j]
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

  x_L = float(l)
  x_U = float(u)
  g_L = float(lb)
  g_U = float(ub)
  model.inner = createProblem(n, x_L, x_U, m, g_L, g_U, nnz, 0,
                              eval_f, eval_g, eval_grad_f, eval_jac_g, nothing)
  model.inner.sense = sense
  addOption(model.inner, "jac_c_constant", "yes")
  addOption(model.inner, "jac_d_constant", "yes")
  addOption(model.inner, "hessian_constant", "yes")
  addOption(model.inner, "hessian_approximation", "limited-memory")
  addOption(model.inner, "mehrotra_algorithm", "yes")
  for (name,value) in model.options
    addOption(model.inner, string(name), value)
  end
end

getsense(m::IpoptMathProgModel) = m.inner.sense
numvar(m::IpoptMathProgModel) = m.inner.n
numconstr(m::IpoptMathProgModel) = m.inner.m
optimize!(m::IpoptMathProgModel) = solveProblem(m.inner)
function status(m::IpoptMathProgModel)
  if m.inner.status == 0 || m.inner.status == 1
    return :Optimal
  end
  return :Infeasible
end
getobjval(m::IpoptMathProgModel) = m.inner.obj_val * (m.inner.sense == :Max ? -1 : +1)
getsolution(m::IpoptMathProgModel) = m.inner.x
getconstrsolution(m::IpoptMathProgModel) = m.inner.g
getreducedcosts(m::IpoptMathProgModel) = zeros(m.inner.n)
getconstrduals(m::IpoptMathProgModel) = zeros(m.inner.m)
getrawsolver(m::IpoptMathProgModel) = m.inner
