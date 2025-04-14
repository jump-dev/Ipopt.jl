# Copyright (c) 2013: Iain Dunning, Miles Lubin, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

using Test

@testset "C" begin
    include("C_wrapper.jl")
end

@testset "MathOptInterface" begin
    include("MOI_wrapper.jl")
end
