module Ipopt

using Libdl

if VERSION < v"1.3" || (
    haskey(ENV, "JULIA_IPOPT_LIBRARY_PATH") &&
    haskey(ENV, "JULIA_IPOPT_EXECUTABLE_PATH")
)
    const _DEPS_FILE = joinpath(dirname(@__DIR__), "deps", "deps.jl")
    if !isfile(_DEPS_FILE)
        error(
            "Ipopt not properly installed. Please run import Pkg; Pkg.build(\"Ipopt\")",
        )
    end
    include(_DEPS_FILE)
else
    import Ipopt_jll: libipopt
end

include("C_wrapper.jl")
include("MOI_wrapper.jl")

end
