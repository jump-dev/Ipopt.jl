module Ipopt

import Ipopt_jll

function __init__()
    global libipopt = Ipopt_jll.libipopt
    return
end

include("C_wrapper.jl")
include("MOI_wrapper.jl")

export IpoptProblem,
    CreateIpoptProblem,
    FreeIpoptProblem,
    AddIpoptStrOption,
    AddIpoptIntOption,
    AddIpoptNumOption,
    OpenIpoptOutputFile,
    SetIpoptProblemScaling,
    SetIntermediateCallback,
    IpoptSolve

end
