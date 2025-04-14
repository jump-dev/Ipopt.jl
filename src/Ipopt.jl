# Copyright (c) 2013: Iain Dunning, Miles Lubin, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module Ipopt

import Ipopt_jll
import Ipopt_jll: libipopt
import LinearAlgebra
import OpenBLAS32_jll
import PrecompileTools  # Needed for MathOptInterfaceExt

function __init__()
    config = LinearAlgebra.BLAS.lbt_get_config()
    if !any(lib -> lib.interface == :lp64, config.loaded_libs)
        LinearAlgebra.BLAS.lbt_forward(OpenBLAS32_jll.libopenblas_path)
    end
    return
end

include("C_wrapper.jl")

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

# These contants are listed here because they are populated by
# MathOptInterfaceExt. They were previously exposed with the `Ipopt.`, and we
# don't want to make a breaking change.
global Optimizer
global CallbackFunction
global column
global _VectorNonlinearOracle

# This function is needed for MOI.SolverVersion, but we don't want to expose
# Ipopt_jll to MathOptInterfaceExt.
_version_string() = string(pkgversion(Ipopt_jll))

end  # module Ipopt
