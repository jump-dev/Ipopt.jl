=================================================
Ipopt.jl --- Julia interface for the Ipopt solver
=================================================

.. module:: Ipopt
   :synopsis: Julia interface for the Ipopt solver

`Ipopt.jl <https://github.com/JuliaOpt/Ipopt.jl>`_ is a light-weight wrapper around
the C interface of `Ipopt <https://projects.coin-or.org/Ipopt>`_, a non-linear
optimizaiton problem solver. You can install it with the Julia package manager::

    julia> Pkg.add("Ipopt")

This will install the interface and Ipopt itself. On Linux, it will build from source
(you will need ``build-essential``, ``gfortran`` and ``liblapack-dev`` installed
on Debian-based systems).
On Windows and OSX, it will download the binary. This document details the Julia interface,
how it relates to the C interface, and any Julia-specific usage notes. For further
information about Ipopt, consult the `official documentation <http://www.coin-or.org/Ipopt/documentation/>`_.

-------
Example
-------

The official documentation uses a sample problem, `HS071 <http://www.coin-or.org/Ipopt/documentation/node20.html>`_, to motivate the various interfaces. Here is what that particular
problem looks like in Julia with the Ipopt.jl interface::

  # HS071
  # min x1 * x4 * (x1 + x2 + x3) + x3
  # st  x1 * x2 * x3 * x4 >= 25
  #     x1^2 + x2^2 + x3^2 + x4^2 = 40
  #     1 <= x1, x2, x3, x4 <= 5
  # Start at (1,5,5,1)
  # End at (1.000..., 4.743..., 3.821..., 1.379...)

  n = 4
  x_L = [1.0, 1.0, 1.0, 1.0]
  x_U = [5.0, 5.0, 5.0, 5.0]

  m = 2
  g_L = [25.0, 40.0]
  g_U = [2.0e19, 40.0]

  function eval_f(x) 
    return x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
  end

  function eval_g(x, g)
    g[1] = x[1]   * x[2]   * x[3]   * x[4]
    g[2] = x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2
  end

  function eval_grad_f(x, grad_f)
    grad_f[1] = x[1] * x[4] + x[4] * (x[1] + x[2] + x[3])
    grad_f[2] = x[1] * x[4]
    grad_f[3] = x[1] * x[4] + 1
    grad_f[4] = x[1] * (x[1] + x[2] + x[3])
  end

  function eval_jac_g(x, mode, rows, cols, values)
    if mode == :Structure
      # Constraint (row) 1
      rows[1] = 1; cols[1] = 1
      rows[2] = 1; cols[2] = 2
      rows[3] = 1; cols[3] = 3
      rows[4] = 1; cols[4] = 4
      # Constraint (row) 2
      rows[5] = 2; cols[5] = 1
      rows[6] = 2; cols[6] = 2
      rows[7] = 2; cols[7] = 3
      rows[8] = 2; cols[8] = 4
    else
      # Constraint (row) 1
      values[1] = x[2]*x[3]*x[4]  # 1,1
      values[2] = x[1]*x[3]*x[4]  # 1,2
      values[3] = x[1]*x[2]*x[4]  # 1,3
      values[4] = x[1]*x[2]*x[3]  # 1,4
      # Constraint (row) 2
      values[5] = 2*x[1]  # 2,1
      values[6] = 2*x[2]  # 2,2
      values[7] = 2*x[3]  # 2,3
      values[8] = 2*x[4]  # 2,4
    end
  end

  function eval_h(x, mode, rows, cols, obj_factor, lambda, values)
    if mode == :Structure
      # Symmetric matrix, fill the lower left triangle only
      idx = 1
      for row = 1:4
        for col = 1:row
          rows[idx] = row
          cols[idx] = col
          idx += 1
        end
      end
    else
      # Again, only lower left triangle
      # Objective
      values[1] = obj_factor * (2*x[4])  # 1,1
      values[2] = obj_factor * (  x[4])  # 2,1
      values[3] = 0                      # 2,2
      values[4] = obj_factor * (  x[4])  # 3,1
      values[5] = 0                      # 3,2
      values[6] = 0                      # 3,3
      values[7] = obj_factor * (2*x[1] + x[2] + x[3])  # 4,1
      values[8] = obj_factor * (  x[1])  # 4,2
      values[9] = obj_factor * (  x[1])  # 4,3
      values[10] = 0                     # 4,4

      # First constraint
      values[2] += lambda[1] * (x[3] * x[4])  # 2,1
      values[4] += lambda[1] * (x[2] * x[4])  # 3,1
      values[5] += lambda[1] * (x[1] * x[4])  # 3,2
      values[7] += lambda[1] * (x[2] * x[3])  # 4,1
      values[8] += lambda[1] * (x[1] * x[3])  # 4,2
      values[9] += lambda[1] * (x[1] * x[2])  # 4,3

      # Second constraint
      values[1]  += lambda[2] * 2  # 1,1
      values[3]  += lambda[2] * 2  # 2,2
      values[6]  += lambda[2] * 2  # 3,3
      values[10] += lambda[2] * 2  # 4,4
    end
  end

  prob = createProblem(n, x_L, x_U, m, g_L, g_U, 8, 10,
                       eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h)

  # Set starting solution
  prob.x = [1.0, 5.0, 5.0, 1.0]

  # Solve
  status = solveProblem(prob)
  
  println(Ipopt.ApplicationReturnStatus[status])
  println(prob.x)
  println(prob.obj_val)

As you can see, the code mirrors the C interface fairly closely, with some C-specific
features abstracted such as replacing the various option-adding functions with one
``addOption`` method.

-----------------
Wrapped Functions
-----------------

We implement all functionality exposed through the C header file ``IpStdCInterface.h``.

createProblem
^^^^^^^^^^^^^

(C function: ``CreateIpoptProblem``)::

  function createProblem(
    n::Int,                     # Number of variables
    x_L::Vector{Float64},       # Variable lower bounds
    x_U::Vector{Float64},       # Variable upper bounds
    m::Int,                     # Number of constraints
    g_L::Vector{Float64},       # Constraint lower bounds
    g_U::Vector{Float64},       # Constraint upper bounds
    nele_jac::Int,              # Number of non-zeros in Jacobian
    nele_hess::Int,             # Number of non-zeros in Hessian
    eval_f,                     # Callback: objective function
    eval_g,                     # Callback: constraint evaluation
    eval_grad_f,                # Callback: objective function gradient
    eval_jac_g,                 # Callback: Jacobian evaluation
    eval_h = nothing)           # Callback: Hessian evaluation

Creates and returns an ``IpoptProblem`` with the given options. Raises error
if something goes wrong during construction. See Callbacks section for more
information about format of callback functions. If you do not provide a callback
for the Hessian, you must set the Hessian approximation option: 
``addOption(prob, "hessian_approximation", "limited-memory")``


freeProblem
^^^^^^^^^^^

(C function: ``FreeIpoptProblem``)::

  function freeProblem(prob::IpoptProblem)

Destroys the internal reference to an ``IpoptProblem``. This function is
automatically called when an ``IpoptProblem`` instance goes out of scope - you
should not need to call it yourself.

addOption
^^^^^^^^^

(C functions: ``AddIpoptStrOption``, ``AddIpoptNumOption``, ``AddIpoptIntOption``)::

  function addOption(
    prob::IpoptProblem, keyword::ASCIIString, value::ASCIIString)

  function addOption(
    prob::IpoptProblem, keyword::ASCIIString, value::Float64)

  function addOption(
    prob::IpoptProblem, keyword::ASCIIString, value::Integer)

Sets a solver option, the full list is available `here <http://www.coin-or.org/Ipopt/documentation/node39.html>`_. Returns nothing, raises error if option could not be set correctly.

openOutputFile
^^^^^^^^^^^^^^

(C function: ``OpenIpoptOutputFile``)::
  
  function openOutputFile(
    prob::IpoptProblem, file_name::ASCIIString, print_level::Int)

Write Ipopt output to a file. Unclear what the acceptable inputs to print
levels are.

setProblemScaling
^^^^^^^^^^^^^^^^^

(C function: ``SetIpoptProblemScaling``)::

  function setProblemScaling(
    prob::IpoptProblem,
    obj_scaling::Float64,       # Objective scaling
    x_scaling = nothing,        # Variable scaling (n-length vector, optional)
    g_scaling = nothing)        # Constraint scaling (m-length vector, optional)

Optional function for scaling the problem. If no input is given for the x and/or
constraint scaling vectors, no scaling is done.

setIntermediateCallback
^^^^^^^^^^^^^^^^^^^^^^^

(C function: ``SetIntermediateCallback``)::

  function setIntermediateCallback(
    prob::IpoptProblem,
    intermediate::Function)

Sets a callback function that will be called after every iteration of the
algorithm. See Callbacks section for more information.

solveProblem
^^^^^^^^^^^^

(C function: ``IpoptSolve``)::

  function solveProblem(prob::IpoptProblem)

  function solveProblem(
    prob::IpoptProblem
    mult_g::Vector{Float64},
    mult_x_L::Vector{Float64},
    mult_x_U::Vector{Float64})

Solves the model created with the above options. Will use the value of ``prob.x``
as the starting point. Stores the final variable values in ``prob.x``, the final
constraint values in ``prob.g``, the final objective in ``prob.obj_value``. The
second version of the function accepts the multipliers on the constraints
and variables bounds and stores the final multipliers in the same vectors. Both
versions return an integer representing the final state. You can access a symbol
representing the meaning of this integer using ``Ipopt.ApplicationReturnStatus``, e.g.::

  status = solveProblem(prob)
  println(Ipopt.ApplicationReturnStatus[status])


---------
Callbacks
---------

All but one of the callbacks for Ipopt evaluate functions given a current solution. The other callback (set by SetIntermediateCallback) receives information from the solver which the user can use as they see fit. This section of the documentation details the function signatures expected for the callbacks. See the HS071 example for full implementations of these for a sample problem.

eval_f
^^^^^^

Returns the value of the objective function at the current solution ``x``::

  function eval_f(x::Vector{Float64})
    # ...
    return obj_value
  end

eval_g
^^^^^^

Sets the value of the constraint functions ``g`` at the current solution ``x``::

  function eval_g(x::Vector{Float64}, g::Vector{Float64})
    # ...
    # g[1] = ...
    # ...
    # g[prob.m] = ...
  end

Note that the values of ``g`` must be set "in-place", i.e. the statement
``g = zeros(prob.m)`` musn't be done. If you do want to create a new vector
and allocate it to ``g`` use ``g[:]``, e.g. ``g[:] = zeros(prob.m)``.

eval_grad_f
^^^^^^^^^^^

Sets the value of the gradient of the objective function at the current solution ``x``::

  function eval_grad_f(x::Vector{Float64}, grad_f::Vector{Float64})
    # ...
    # grad_f[1] = ...
    # ...
    # grad_f[prob.n] = ...
  end

As for ``eval_g``, you must set the values "in-place".

eval_jac_g
^^^^^^^^^^

This function has two modes of operation. In the first mode the user tells IPOPT the sparsity structure of the Jacobian of the constraints. In the second mode the user provides the actual Jacobian values. Julia is 1-based, in the sense that indexing always starts at 1 (unlike C, which starts at 0).::

  function eval_jac_g(    
    x::Vector{Float64},         # Current solution
    mode,                       # Either :Structure or :Values
    rows::Vector{Int32},        # Sparsity structure - row indices
    cols::Vector{Int32},        # Sparsity structure - column indices
    values::Vector{Float64})    # The values of the Hessian

    if mode == :Structure
      # rows[...] = ...
      # ...
      # cols[...] = ...
    else
      # values[...] = ...
    end
  end

As for the previous two callbacks, all values must be set "in-place". See the Ipopt documentation for a further description of the sparsity format followed by Ipopt ((row,column,value) triples).

eval_h
^^^^^^

Similar to the Jacobian, except for the Hessian of the Lagrangian. See documentation for full details of the meaning of everything.::

  function eval_h(       
    x::Vector{Float64},         # Current solution
    mode,                       # Either :Structure or :Values
    rows::Vector{Int32},        # Sparsity structure - row indices
    cols::Vector{Int32},        # Sparsity structure - column indices
    obj_factor::Float64,        # Lagrangian multiplier for objective
    lambda::Vector{Float64},    # Multipliers for each constraint
    values::Vector{Float64})    # The values of the Hessian

    if mode == :Structure
      # rows[...] = ...
      # ...
      # cols[...] = ...
    else
      # values[...] = ...
    end
  end

This function does not need to be provided - see createProblem for more information.

intermediate
^^^^^^^^^^^^

Different from the above, this function is called every iteration and allows the user to track the progress of the solve. Additionally they can terminate the optimization prematurely. Must return true (keep going) or false (stop).::

  function intermediate(
    alg_mod::Int,
    iter_count::Int, 
    obj_value::Float64,
    inf_pr::Float64, inf_du::Float64,
    mu::Float64, d_norm::Float64,
    regularization_size::Float64,
    alpha_du::Float64, alpha_pr::Float64, 
    ls_trials::Int)
    # ...
    return true  # Keep going
  end

For descriptions of inputs, see official documentation.
