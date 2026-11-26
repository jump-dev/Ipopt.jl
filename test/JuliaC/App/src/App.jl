# Copyright (c) 2013: Iain Dunning, Miles Lubin, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module App

import Ipopt

struct HS071 <: Ipopt.AbstractOracle
    n::Int
    x_L::Vector{Float64}
    x_U::Vector{Float64}
    m::Int
    g_L::Vector{Float64}
    g_U::Vector{Float64}

    function HS071()
        return new(
            4,
            [1.0, 1.0, 1.0, 1.0],
            [5.0, 5.0, 5.0, 5.0],
            2,
            [25.0, 40.0],
            [2.0e19, 40.0],
        )
    end
end

function Ipopt.eval_f(::HS071, x::Vector{Float64})
    return x[1] * x[4] * (x[1] + x[2] + x[3]) + x[3]
end

function Ipopt.eval_g(::HS071, x::Vector{Float64}, g::Vector{Float64})
    g[1] = x[1] * x[2] * x[3] * x[4]
    g[2] = x[1]^2 + x[2]^2 + x[3]^2 + x[4]^2
    return
end

function Ipopt.eval_grad_f(::HS071, x::Vector{Float64}, grad_f::Vector{Float64})
    grad_f[1] = x[1] * x[4] + x[4] * (x[1] + x[2] + x[3])
    grad_f[2] = x[1] * x[4]
    grad_f[3] = x[1] * x[4] + 1
    grad_f[4] = x[1] * (x[1] + x[2] + x[3])
    return
end

function Ipopt.eval_jac_g(
    ::HS071,
    x::Vector{Float64},
    rows::Vector{Int32},
    cols::Vector{Int32},
    values::Union{Nothing,Vector{Float64}},
)
    if values === nothing
        rows[1], cols[1] = 1, 1
        rows[2], cols[2] = 1, 2
        rows[3], cols[3] = 1, 3
        rows[4], cols[4] = 1, 4
        rows[5], cols[5] = 2, 1
        rows[6], cols[6] = 2, 2
        rows[7], cols[7] = 2, 3
        rows[8], cols[8] = 2, 4
    else
        values[1] = x[2] * x[3] * x[4]
        values[2] = x[1] * x[3] * x[4]
        values[3] = x[1] * x[2] * x[4]
        values[4] = x[1] * x[2] * x[3]
        values[5] = 2 * x[1]
        values[6] = 2 * x[2]
        values[7] = 2 * x[3]
        values[8] = 2 * x[4]
    end
    return
end

Ipopt.has_eval_h(::HS071) = true

function Ipopt.eval_h(
    ::HS071,
    x::Vector{Float64},
    rows::Vector{Int32},
    cols::Vector{Int32},
    σ::Float64,
    lambda::Vector{Float64},
    values::Union{Nothing,Vector{Float64}},
)
    if values === nothing
        idx = 1
        for row in 1:4
            for col in 1:row
                rows[idx] = row
                cols[idx] = col
                idx += 1
            end
        end
    else
        values[1] = 2 * (σ * x[4] + lambda[2])
        values[2] = x[4] * (σ + lambda[1] * x[3])
        values[3] = 2 * lambda[2]
        values[4] = x[4] * (σ + lambda[1] * x[2])
        values[5] = lambda[1] * x[1] * x[4]
        values[6] = 2 * lambda[2]
        values[7] = σ * (2 * x[1] + x[2] + x[3]) + lambda[1] * x[2] * x[3]
        values[8] = x[1] * (σ + lambda[1] * x[3])
        values[9] = x[1] * (σ + lambda[1] * x[2])
        values[10] = 2 * lambda[2]
    end
    return
end

function Ipopt.eval_intermediate(
    prob::Ipopt.Problem{HS071},
    oracle::HS071,
    alg_mod::Cint,
    iter_count::Cint,
    obj_value::Float64,
    inf_pr::Float64,
    inf_du::Float64,
    mu::Float64,
    d_norm::Float64,
    regularization_size::Float64,
    alpha_du::Float64,
    alpha_pr::Float64,
    ls_trials::Cint,
)
    m, n = oracle.m, oracle.n
    x, z_L, z_U = zeros(n), zeros(n), zeros(n)
    g, lambda = zeros(m), zeros(m)
    scaled = false
    Ipopt.GetIpoptCurrentIterate(prob, scaled, n, x, z_L, z_U, m, g, lambda)
    x_L_violation, x_U_violation = zeros(n), zeros(n)
    compl_x_L, compl_x_U, grad_lag_x = zeros(n), zeros(n), zeros(n)
    nlp_constraint_violation, compl_g = zeros(m), zeros(m)
    Ipopt.GetIpoptCurrentViolations(
        prob,
        scaled,
        n,
        x_L_violation,
        x_U_violation,
        compl_x_L,
        compl_x_U,
        grad_lag_x,
        m,
        nlp_constraint_violation,
        compl_g,
    )
    violation =
        sum(x_L_violation) + sum(x_U_violation) + sum(nlp_constraint_violation)
    Core.println(Core.stdout, "Violation = ", violation)
    return true
end

function @main(ARGS)
    oracle = HS071()
    prob = Ipopt.CreateIpoptProblem(
        oracle.n,
        oracle.x_L,
        oracle.x_U,
        oracle.m,
        oracle.g_L,
        oracle.g_U,
        8,
        10,
        oracle,
    )
    prob.x = [1.0, 5.0, 5.0, 1.0]
    Ipopt.SetIntermediateCallback(prob)
    Ipopt.IpoptSolve(prob)
    return 0
end

end  # App
