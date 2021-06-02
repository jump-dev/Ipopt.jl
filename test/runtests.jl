if get(ENV, "GITHUB_ACTIONS", "") == "true"
    import Pkg
    Pkg.add(Pkg.PackageSpec(name = "MathOptInterface", rev = "master"))
end

using Test

function runtests(mod)
    for name in names(mod; all = true)
        if !startswith("$(name)", "test_")
            continue
        end
        @testset "$(name)" begin
            getfield(mod, name)()
        end
    end
end

@testset "C" begin
    include("C_wrapper.jl")
end

@testset "MathOptInterface" begin
    include("MOI_wrapper.jl")
end

@testset "MathProgBase" begin
    include("MPB_wrapper.jl")
end
