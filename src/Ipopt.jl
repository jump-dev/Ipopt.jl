# Copyright (c) 2013: Iain Dunning, Miles Lubin, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

module Ipopt

import Ipopt_jll
import Ipopt_jll: libipopt
import LinearAlgebra
import OpenBLAS32_jll
import PrecompileTools  # Needed for IpoptMathOptInterfaceExt

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

# This function is needed for MOI.SolverVersion, but we don't want to expose
# Ipopt_jll to IpoptMathOptInterfaceExt.
function _version_string()
    jll_version = pkgversion(Ipopt_jll)
    if jll_version ≥ v"300.1400.1900"
        major = Ref{Cint}()
        minor = Ref{Cint}()
        patch = Ref{Cint}()
        @ccall libipopt.GetIpoptVersion(major::Ptr{Cint}, minor::Ptr{Cint}, patch::Ptr{Cint})::Cvoid
        return string(major[], ".", minor[], ".", patch[])
    else
        return string(jll_version)
    end
end

# This function is needed by the MOI wrapper. It was previously exposed as
# Ipopt.column, so we keep it here for backwards compatibility.
function column end

global Optimizer
global CallbackFunction
global _VectorNonlinearOracle

end  # module Ipopt
