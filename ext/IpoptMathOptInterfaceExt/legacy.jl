# Copyright (c) 2013: Iain Dunning, Miles Lubin, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

# Support for the legacy MOI.NLPBlock interface. This file can be removed when
# that interface is removed from MOI.

_objective_sign(sense::MOI.OptimizationSense) =
    sense == MOI.MAX_SENSE ? -1.0 : sense == MOI.MIN_SENSE ? 1.0 : 0.0

struct _NLPBlockEvaluator <: MOI.AbstractNLPEvaluator
    data::MOI.NLPBlockData
    sense::MOI.OptimizationSense
end

MOI.features_available(d::_NLPBlockEvaluator) =
    MOI.features_available(d.data.evaluator)
MOI.initialize(d::_NLPBlockEvaluator, features) =
    MOI.initialize(d.data.evaluator, features)
MOI.Nonlinear._constraint_bounds(d::_NLPBlockEvaluator) =
    d.data.constraint_bounds
MOI.eval_objective(d::_NLPBlockEvaluator, x) =
    _objective_sign(d.sense) * MOI.eval_objective(d.data.evaluator, x)
function MOI.eval_objective_gradient(d::_NLPBlockEvaluator, g, x)
    MOI.eval_objective_gradient(d.data.evaluator, g, x)
    g .*= _objective_sign(d.sense)
    return
end
MOI.eval_constraint(d::_NLPBlockEvaluator, g, x) =
    MOI.eval_constraint(d.data.evaluator, g, x)
MOI.jacobian_structure(d::_NLPBlockEvaluator) =
    MOI.jacobian_structure(d.data.evaluator)
MOI.eval_constraint_jacobian(d::_NLPBlockEvaluator, J, x) =
    MOI.eval_constraint_jacobian(d.data.evaluator, J, x)
MOI.hessian_lagrangian_structure(d::_NLPBlockEvaluator) =
    MOI.hessian_lagrangian_structure(d.data.evaluator)
function MOI.eval_hessian_lagrangian(d::_NLPBlockEvaluator, H, x, σ, μ)
    sign = _objective_sign(d.sense)
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
        model.model.objective_sink = MOI.Nonlinear._INNER
    end
    oracles = model.model.inner
    inner = MOI.Nonlinear.EvaluatorWithOracles(
        oracles,
        _NLPBlockEvaluator(
            model.nlp_data,
            MOI.get(model.model, MOI.ObjectiveSense()),
        ),
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
MOI.get(model::Optimizer, ::MOI.NLPBlock) = model.nlp_data
function MOI.set(model::Optimizer, ::MOI.NLPBlock, nlp_data::MOI.NLPBlockData)
    if model.number_of_nonlinear_constraints > 0 || model.has_nonlinear_objective
        error("Cannot mix the new and legacy nonlinear APIs")
    end
    model.nlp_data = nlp_data
    model.uses_nlp_block = !(nlp_data.evaluator isa _EmptyNLPEvaluator)
    if model.uses_nlp_block
        model.has_constant_constraint_jacobian = false
        model.has_constant_constraint_hessian = false
        model.has_constant_objective_hessian = false
    end
    model.inner = nothing
    return
end

function MOI.get(model::Optimizer, attr::MOI.NLPBlockDual)
    MOI.check_result_index_bounds(model, attr)
    offset = length(model.inner.mult_g) - length(model.nlp_data.constraint_bounds)
    return -model.inner.mult_g[(offset+1):end]
end
