using Ipopt, Base.Test
using Compat

@testset "C API" begin
    # First of all, test that hs071 example works
    include("hs071_test.jl")

    # Test setting some options
    # String option
    println("\nString option")
    addOption(prob, "hessian_approximation", "exact")
    @test_throws ErrorException addOption(prob, "hessian_approximation", "badoption")
    println("\nInt option")
    # Int option
    addOption(prob, "file_print_level", 3) == nothing
    @test_throws ErrorException addOption(prob, "file_print_level", -1)
    # Double option
    println("\nFloat option")
    addOption(prob, "derivative_test_tol", 0.5)
    @test_throws ErrorException addOption(prob, "derivative_test_tol", -1.0)

    # Test opening an output file
    openOutputFile(prob, "blah.txt", 5)
    Ipopt.freeProblem(prob) # Needed before the `rm` on Windows.
    # unlink the output file
    rm("blah.txt")

    # Test that the ipopt binary works
    # See https://github.com/JuliaOpt/Ipopt.jl/issues/119 for discussion of the
    # known failure on Windows and Julia 0.7.
    if !(Compat.Sys.iswindows() && VERSION >= v"0.7-")
        @test success(`$(Ipopt.amplexe) -v`)
    end
end

@testset "MathProgBase" begin
    include("MPBWrapper.jl")
end

@testset "MathOptInterface" begin
    include("MOIWrapper.jl")
end
