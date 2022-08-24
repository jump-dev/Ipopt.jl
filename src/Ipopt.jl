# Copyright (c) 2013: Iain Dunning, Miles Lubin, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module Ipopt

import Ipopt_jll
import OpenBLAS32_jll
import MathOptInterface
import LinearAlgebra

const MOI = MathOptInterface

function __init__()
    (VERSION ≥ v"1.8") && LinearAlgebra.BLAS.lbt_forward(OpenBLAS32_jll.libopenblas_path)
    global libipopt = Ipopt_jll.libipopt
    return
end

include("C_wrapper.jl")
include("MOI_wrapper.jl")

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
