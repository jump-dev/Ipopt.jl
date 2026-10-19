# Copyright (c) 2013: Iain Dunning, Miles Lubin, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

# Support for the legacy MOI.NLPBlock interface. This file can be removed when
# that interface is removed from MOI.

function _objective_sign(sense::MOI.OptimizationSense)
    return sense == MOI.MAX_SENSE ? -1.0 : sense == MOI.MIN_SENSE ? 1.0 : 0.0
end

# The objective sense is read from `model` at each evaluation because it may
# be modified after the evaluator is created.
struct _NLPBlockEvaluator{M<:MOI.ModelLike} <: MOI.AbstractNLPEvaluator
    data::MOI.NLPBlockData
    model::M
end

function _objective_sign(d::_NLPBlockEvaluator)
    return _objective_sign(MOI.get(d.model, MOI.ObjectiveSense()))
end

function MOI.features_available(d::_NLPBlockEvaluator)
    return MOI.features_available(d.data.evaluator)
end
function MOI.initialize(d::_NLPBlockEvaluator, features)
    return MOI.initialize(d.data.evaluator, features)
end
function MOI.eval_objective(d::_NLPBlockEvaluator, x)
    return _objective_sign(d) * MOI.eval_objective(d.data.evaluator, x)
end
function MOI.eval_objective_gradient(d::_NLPBlockEvaluator, g, x)
    MOI.eval_objective_gradient(d.data.evaluator, g, x)
    g .*= _objective_sign(d)
    return
end

# Return all constraint bounds in evaluator row order. The first part is the
# regular model stack; a legacy NLP block, when present, is appended by the
# legacy evaluator at the innermost position.
function _constraint_bounds(model::Optimizer)
    bounds = MOI.Utilities.constraint_bounds(model.model)
    if !model.uses_nlp_block
        return bounds
    end
    legacy = model.nlp_data.constraint_bounds
    lower = vcat(bounds.lower, getfield.(legacy, :lower))
    upper = vcat(bounds.upper, getfield.(legacy, :upper))
    return MOI.Utilities.Hyperrectangle(lower, upper)
end

# A block without constraints is not required to support the Jacobian, for
# instance, it may not have the `:Jac` feature.
_has_constraints(d::_NLPBlockEvaluator) = !isempty(d.data.constraint_bounds)

function MOI.eval_constraint(d::_NLPBlockEvaluator, g, x)
    if _has_constraints(d)
        MOI.eval_constraint(d.data.evaluator, g, x)
    end
    return
end

function MOI.jacobian_structure(d::_NLPBlockEvaluator)
    if !_has_constraints(d)
        return Tuple{Int64,Int64}[]
    end
    return MOI.jacobian_structure(d.data.evaluator)
end

function MOI.eval_constraint_jacobian(d::_NLPBlockEvaluator, J, x)
    if _has_constraints(d)
        MOI.eval_constraint_jacobian(d.data.evaluator, J, x)
    end
    return
end
function MOI.hessian_lagrangian_structure(d::_NLPBlockEvaluator)
    return MOI.hessian_lagrangian_structure(d.data.evaluator)
end
function MOI.eval_hessian_lagrangian(d::_NLPBlockEvaluator, H, x, σ, μ)
    sign = _objective_sign(d)
    return MOI.eval_hessian_lagrangian(d.data.evaluator, H, x, sign * σ, μ)
end

function _legacy_evaluator(model::Optimizer, vars)
    if !(model.model isa MOI.Nonlinear.ModelWithQuad)
        error(
            "The legacy `MOI.NLPBlock` interface cannot be combined " *
            "with the selected automatic-differentiation backend.",
        )
    end
    if model.nlp_data.has_objective
        # The legacy NLPBlock path only supports the built-in layered model,
        # not backend-specific models such as ExaModels.
        @assert model.model isa MOI.Nonlinear.ModelWithQuad
        model.model.objective_sink = MOI.Nonlinear._INNER
    end
    oracles = model.model.inner
    inner = MOI.Nonlinear.EvaluatorWithOracles(
        oracles,
        _NLPBlockEvaluator(model.nlp_data, model.model),
        vars,
    )
    return MOI.Nonlinear.EvaluatorWithQuad(model.model, inner)
end

MOI.supports(::Optimizer, ::MOI.NLPBlockDualStart) = true
function MOI.set(
    model::Optimizer,
    ::MOI.NLPBlockDualStart,
    values::Union{Nothing,Vector},
)
    model.nlp_dual_start = values
    return
end
MOI.get(model::Optimizer, ::MOI.NLPBlockDualStart) = model.nlp_dual_start

MOI.supports(::Optimizer, ::MOI.NLPBlock) = true

# After `optimize!`, this also returns the block created from the model, for
# instance from its `ScalarNonlinearFunction`s.
function MOI.get(model::Optimizer, ::MOI.NLPBlock)
    if model.uses_nlp_block || model.inner === nothing
        return model.nlp_data
    end
    return MOI.NLPBlockData(model.evaluator)
end

function MOI.set(model::Optimizer, ::MOI.NLPBlock, nlp_data::MOI.NLPBlockData)
    P = MOI.Parameter{Float64}
    if model.number_of_nonlinear_constraints > 0 ||
       model.has_nonlinear_objective ||
       MOI.get(model.model, MOI.NumberOfConstraints{MOI.VariableIndex,P}()) > 0
        error("Cannot mix the new and legacy nonlinear APIs")
    end
    model.nlp_data = nlp_data
    model.uses_nlp_block = !(nlp_data.evaluator isa _EmptyNLPEvaluator)
    if model.uses_nlp_block
        model.has_constant_constraint_jacobian = false
        model.has_constant_objective_hessian = false
    end
    model.inner = nothing
    return
end

function MOI.get(model::Optimizer, attr::MOI.NLPBlockDual)
    MOI.check_result_index_bounds(model, attr)
    offset =
        length(model.inner.mult_g) - length(model.nlp_data.constraint_bounds)
    return -model.inner.mult_g[(offset+1):end]
end
