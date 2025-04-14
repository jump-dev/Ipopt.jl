# Copyright (c) 2013: Iain Dunning, Miles Lubin, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module Ipopt

import Ipopt_jll
import LinearAlgebra
import OpenBLAS32_jll

function __init__()
    if VERSION >= v"1.9"
        config = LinearAlgebra.BLAS.lbt_get_config()
        if !any(lib -> lib.interface == :lp64, config.loaded_libs)
            LinearAlgebra.BLAS.lbt_forward(OpenBLAS32_jll.libopenblas_path)
        end
    end
    global libipopt = Ipopt_jll.libipopt
    return
end

include("C_wrapper.jl")

"""
    Optimizer()

Create a new Ipopt optimizer.
"""
function Optimizer end

export IpoptProblem,
    CreateIpoptProblem,
    FreeIpoptProblem,
    AddIpoptStrOption,
    AddIpoptNumOption,
    AddIpoptIntOption,
    OpenIpoptOutputFile,
    SetIpoptProblemScaling,
    SetIntermediateCallback,
    IpoptSolve
end
