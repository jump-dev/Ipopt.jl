=================================================
Ipopt.jl --- Julia interface for the Ipopt solver
=================================================

.. module:: Ipopt
   :synopsis: Julia interface for the Ipopt solver

`Ipopt.jl <https://github.com/mlubin/Ipopt.jl>`_ is a light-weight wrapper around
the C interface of `Ipopt <https://projects.coin-or.org/Ipopt>`_, a non-linear
optimizaiton problem solver. You can install it with the Julia package manager::

    julia> Pkg.add("Ipopt")

This document details the Julia interface, how it relates to the C interface, and
any Julia-specific usage notes. For further information about Ipopt, consult the
`official documentation <http://www.coin-or.org/Ipopt/documentation/>`_.

-------
Example
-------

The official documentation uses a sample problem, `HS071 <http://www.coin-or.org/Ipopt/documentation/node20.html>`_, to motivate the various interfaces. Here is what that particular
problem looks like in Julia with the Ipopt.jl interface::

  n = 4
  x_L = [1.0, 1.0, 1.0, 1.0]
  x_U = [5.0, 5.0, 5.0, 5.0]

  m = 2
  g_L = [25.0, 40.0]
  g_U = [2.0e19, 40.0]

  prob = CreateProblem(n, x_L, x_U, m, g_L, g_U, 8, 10,
                       eval_f, eval_g, eval_grad_f, eval_jac_g, eval_h)

  AddOption(prob, "hessian_approximation", "limited-memory")

  prob.x = [1.0, 5.0, 5.0, 1.0]
  status = SolveProblem(prob)
  
  println(Ipopt.ApplicationReturnStatus[status])
  println(prob.x)
  println(prob.obj_val)

As you can see, the code mirrors the C interface fairly closely, with some C-specific
features abstracted such as replacing the various option-adding functions with one
``AddOption`` method.

---------
Functions
---------

We implement all functionality exposed through the C header file ``IpStdCInterface.h``.

CreateProblem
^^^^^^^^^^^^^

(C function(s): ``CreateIpoptProblem``)::

  function CreateProblem(
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
    eval_h)                     # Callback: Hessian evaluation

Creates and returns an ``IpoptProblem`` with the given options. Raises error
if something goes wrong during construction. See Callbacks section for more
information about callback functions.

FreeProblem
^^^^^^^^^^^

(C function(s): ``FreeIpoptProblem``)::

  function FreeProblem(prob::IpoptProblem)

Destroys the internal reference to an ``IpoptProblem``. This function is
automatically called when an ``IpoptProblem`` instance goes out of scope - you
should not need to call it yourself.

AddOption
^^^^^^^^^

(C functions(s): ``AddIpoptStrOption``, ``AddIpoptNumOption``, ``AddIpoptIntOption``)::

  function AddOption(
    prob::IpoptProblem, keyword::ASCIIString, value::ASCIIString)

  function AddOption(
    prob::IpoptProblem, keyword::ASCIIString, value::Float64)

  function AddOption(
    prob::IpoptProblem, keyword::ASCIIString, value::Integer)

Sets a solver option, the full list is available `here <http://www.coin-or.org/Ipopt/documentation/node39.html>`_. Returns nothing, raises error if option could not be set correctly.

OpenOutputFile
^^^^^^^^^^^^^^

(C function(s): ``OpenIpoptOutputFile``)::
  
  function OpenOutputFile(
    prob::IpoptProblem, file_name::ASCIIString, print_level::Int)

Write Ipopt output to a file. Unclear what the acceptable inputs to print
levels are.

SetProblemScaling
^^^^^^^^^^^^^^^^^

(C function(s): ``SetIpoptProblemScaling``)::

  function SetProblemScaling(
    prob::IpoptProblem,
    obj_scaling::Float64,       # Objective scaling
    x_scaling = nothing,        # Variable scaling (n-length vector, optional)
    g_scaling = nothing)        # Constraint scaling (m-length vector, optional)

Optional function for scaling the problem. If no input is given for the x and/or
constraint scaling vectors, no scaling is done.
