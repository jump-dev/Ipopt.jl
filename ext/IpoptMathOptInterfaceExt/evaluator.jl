"""
    _VectorNonlinearOracle(;
        dimension::Int,
        l::Vector{Float64},
        u::Vector{Float64},
        eval_f::Function,
        jacobian_structure::Vector{Tuple{Int,Int}},
        eval_jacobian::Function,
        hessian_lagrangian_structure::Vector{Tuple{Int,Int}} = Tuple{Int,Int}[],
        eval_hessian_lagrangian::Union{Nothing,Function} = nothing,
    ) <: MOI.AbstractVectorSet

The set:
```math
S = \\{x \\in \\mathbb{R}^{dimension}: l \\le f(x) \\le u\\}
```
where ``f`` is defined by the vectors `l` and `u`, and the callback oracles
`eval_f`, `eval_jacobian`, and `eval_hessian_lagrangian`.

!!! warning
    This set is experimental. We will decide by September 30, 2025, whether to
    convert this into the public `Ipopt.VectorNonlinearOracle`, move it to
    `MOI.VectorNonlinearOracle`, or remove it completely.

## f

The `eval_f` function must have the signature
```julia
eval_f(ret::AbstractVector, x::AbstractVector)::Nothing
```
which fills ``f(x)`` into the dense vector `ret`.

## Jacobian

The `eval_jacobian` function must have the signature
```julia
eval_jacobian(ret::AbstractVector, x::AbstractVector)::Nothing
```
which fills the sparse Jacobian ``\\nabla f(x)`` into `ret`.

The one-indexed sparsity structure must be provided in the `jacobian_structure`
argument.

## Hessian

The `eval_hessian_lagrangian` function is optional.

If `eval_hessian_lagrangian === nothing`, Ipopt will use a Hessian approximation
instead of the exact Hessian.

If `eval_hessian_lagrangian` is a function, it must have the signature
```julia
eval_hessian_lagrangian(
    ret::AbstractVector,
    x::AbstractVector,
    μ::AbstractVector,
)::Nothing
```
which fills the sparse Hessian of the Lagrangian ``\\sum \\mu_i \\nabla^2 f_i(x)``
into `ret`.

The one-indexed sparsity structure must be provided in the
`hessian_lagrangian_structure` argument.

## Example

To model the set:
```math
\\begin{align}
0 \\le & x^2           \\le 1
0 \\le & y^2 + z^3 - w \\le 0
\\end{align}
```
do
```jldoctest
julia> import Ipopt

julia> set = Ipopt._VectorNonlinearOracle(;
           dimension = 3,
           l = [0.0, 0.0],
           u = [1.0, 0.0],
           eval_f = (ret, x) -> begin
               ret[1] = x[2]^2
               ret[2] = x[3]^2 + x[4]^3 - x[1]
               return
           end,
           jacobian_structure = [(1, 2), (2, 1), (2, 3), (2, 4)],
           eval_jacobian = (ret, x) -> begin
               ret[1] = 2.0 * x[2]
               ret[2] = -1.0
               ret[3] = 2.0 * x[3]
               ret[4] = 3.0 * x[4]^2
               return
           end,
           hessian_lagrangian_structure = [(2, 2), (3, 3), (4, 4)],
           eval_hessian_lagrangian = (ret, x, u) -> begin
               ret[1] = 2.0 * u[1]
               ret[2] = 2.0 * u[2]
               ret[3] = 6.0 * x[4] * u[2]
               return
           end,
       );
```
"""
struct _VectorNonlinearOracle <: MOI.AbstractVectorSet
    input_dimension::Int
    output_dimension::Int
    l::Vector{Float64}
    u::Vector{Float64}
    eval_f::Function
    jacobian_structure::Vector{Tuple{Int,Int}}
    eval_jacobian::Function
    hessian_lagrangian_structure::Vector{Tuple{Int,Int}}
    eval_hessian_lagrangian::Union{Nothing,Function}
    # Temporary storage
    x::Vector{Float64}

    function _VectorNonlinearOracle(;
        dimension::Int,
        l::Vector{Float64},
        u::Vector{Float64},
        eval_f::Function,
        jacobian_structure::Vector{Tuple{Int,Int}},
        eval_jacobian::Function,
        # The hessian_lagrangian is optional.
        hessian_lagrangian_structure::Vector{Tuple{Int,Int}} = Tuple{Int,Int}[],
        eval_hessian_lagrangian::Union{Nothing,Function} = nothing,
    )
        @assert length(l) == length(u)
        return new(
            dimension,
            length(l),
            l,
            u,
            eval_f,
            jacobian_structure,
            eval_jacobian,
            hessian_lagrangian_structure,
            eval_hessian_lagrangian,
            # Temporary storage
            zeros(dimension),
        )
    end
end

MOI.dimension(s::_VectorNonlinearOracle) = s.input_dimension

MOI.copy(s::_VectorNonlinearOracle) = s


struct EvaluatorWithQuad{T,E}
    has_objective::Bool
    has_nlp_objective::Bool
    has_nlp_constraints::Bool
    qp::QPBlockData{T}
    vector_nonlinear_oracle_constraints::Vector{
        Tuple{MOI.VectorOfVariables,_VectorNonlinearOracle},
    }
    nlp::E
end

### Eval_F_CB

function MOI.eval_objective(evaluator::EvaluatorWithQuad, x)
    # TODO(odow): FEASIBILITY_SENSE could produce confusing solver output if
    # a nonzero objective is set.
    if !evaluator.has_objective
        return 0.0
    elseif evaluator.has_nlp_objective
        return MOI.eval_objective(evaluator.nlp, x)::Float64
    end
    return MOI.eval_objective(evaluator.qp, x)
end

### Eval_Grad_F_CB

function MOI.eval_objective_gradient(evaluator::EvaluatorWithQuad, grad, x)
    if !evaluator.has_objective
        grad .= zero(eltype(grad))
    elseif evaluator.has_nlp_objective
        MOI.eval_objective_gradient(evaluator.nlp, grad, x)
    else
        MOI.eval_objective_gradient(evaluator.qp, grad, x)
    end
    return
end

### Eval_G_CB

function _eval_constraint(
    g::AbstractVector,
    offset::Int,
    x::AbstractVector,
    f::MOI.VectorOfVariables,
    s::_VectorNonlinearOracle,
)
    for i in 1:s.input_dimension
        s.x[i] = x[f.variables[i].value]
    end
    ret = view(g, offset .+ (1:s.output_dimension))
    s.eval_f(ret, s.x)
    return offset + s.output_dimension
end

function MOI.eval_constraint(evaluator::EvaluatorWithQuad, g, x)
    MOI.eval_constraint(evaluator.qp, g, x)
    offset = length(evaluator.qp)
    for (f, s) in evaluator.vector_nonlinear_oracle_constraints
        offset = _eval_constraint(g, offset, x, f, s)
    end
    g_nlp = view(g, (offset+1):length(g))
    MOI.eval_constraint(evaluator.nlp, g_nlp, x)
    return
end

### Eval_Jac_G_CB

function _jacobian_structure(
    ret::AbstractVector,
    row_offset::Int,
    f::MOI.VectorOfVariables,
    s::_VectorNonlinearOracle,
)
    for (i, j) in s.jacobian_structure
        push!(ret, (row_offset + i, f.variables[j].value))
    end
    return row_offset + s.output_dimension
end

function MOI.jacobian_structure(evaluator::EvaluatorWithQuad)
    J = MOI.jacobian_structure(evaluator.qp)
    offset = length(evaluator.qp)
    for (f, s) in evaluator.vector_nonlinear_oracle_constraints
        offset = _jacobian_structure(J, offset, f, s)
    end
    if evaluator.has_nlp_constraints
        J_nlp = MOI.jacobian_structure(
            evaluator.nlp,
        )::Vector{Tuple{Int64,Int64}}
        for (row, col) in J_nlp
            push!(J, (row + offset, col))
        end
    end
    return J
end

function _eval_constraint_jacobian(
    values::AbstractVector,
    offset::Int,
    x::AbstractVector,
    f::MOI.VectorOfVariables,
    s::_VectorNonlinearOracle,
)
    for i in 1:s.input_dimension
        s.x[i] = x[f.variables[i].value]
    end
    nnz = length(s.jacobian_structure)
    s.eval_jacobian(view(values, offset .+ (1:nnz)), s.x)
    return offset + nnz
end

function MOI.eval_constraint_jacobian(evaluator::EvaluatorWithQuad, values, x)
    offset = MOI.eval_constraint_jacobian(evaluator.qp, values, x)
    offset -= 1  # .qp_data returns one-indexed offset
    for (f, s) in evaluator.vector_nonlinear_oracle_constraints
        offset = _eval_constraint_jacobian(values, offset, x, f, s)
    end
    nlp_values = view(values, (offset+1):length(values))
    MOI.eval_constraint_jacobian(evaluator.nlp, nlp_values, x)
    return
end

### Eval_H_CB

function _hessian_lagrangian_structure(
    ret::AbstractVector,
    f::MOI.VectorOfVariables,
    s::_VectorNonlinearOracle,
)
    for (i, j) in s.hessian_lagrangian_structure
        push!(ret, (f.variables[i].value, f.variables[j].value))
    end
    return
end

function MOI.hessian_lagrangian_structure(evaluator::EvaluatorWithQuad)
    H = MOI.hessian_lagrangian_structure(evaluator.qp)
    for (f, s) in evaluator.vector_nonlinear_oracle_constraints
        _hessian_lagrangian_structure(H, f, s)
    end
    append!(H, MOI.hessian_lagrangian_structure(evaluator.nlp))
    return H
end

function _eval_hessian_lagrangian(
    H::AbstractVector,
    H_offset::Int,
    x::AbstractVector,
    μ::AbstractVector,
    μ_offset::Int,
    f::MOI.VectorOfVariables,
    s::_VectorNonlinearOracle,
)
    for i in 1:s.input_dimension
        s.x[i] = x[f.variables[i].value]
    end
    H_nnz = length(s.hessian_lagrangian_structure)
    H_view = view(H, H_offset .+ (1:H_nnz))
    μ_view = view(μ, μ_offset .+ (1:s.output_dimension))
    s.eval_hessian_lagrangian(H_view, s.x, μ_view)
    return H_offset + H_nnz, μ_offset + s.output_dimension
end

function MOI.eval_hessian_lagrangian(evaluator::EvaluatorWithQuad, H, x, σ, μ)
    offset = MOI.eval_hessian_lagrangian(evaluator.qp, H, x, σ, μ)
    offset -= 1  # .qp_data returns one-indexed offset
    μ_offset = length(evaluator.qp)
    for (f, s) in evaluator.vector_nonlinear_oracle_constraints
        offset, μ_offset =
            _eval_hessian_lagrangian(H, offset, x, μ, μ_offset, f, s)
    end
    H_nlp = view(H, (offset+1):length(H))
    μ_nlp = view(μ, (μ_offset+1):length(μ))
    MOI.eval_hessian_lagrangian(evaluator.nlp, H_nlp, x, σ, μ_nlp)
    return
end
