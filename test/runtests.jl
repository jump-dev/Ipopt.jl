# Copyright (c) 2013: Iain Dunning, Miles Lubin, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

import Pkg
if get(ENV, "CI", "false") == "true"
    Pkg.pkg"add MathOptInterface#od/nlp-expr"
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
