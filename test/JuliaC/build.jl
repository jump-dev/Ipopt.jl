# Copyright (c) 2013: Iain Dunning, Miles Lubin, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

using Test

import JuliaC

output_dir = joinpath(@__DIR__, "build")
rm(output_dir; force = true, recursive = true)

outname = joinpath(output_dir, "app_test_exe")

image_recipe = JuliaC.ImageRecipe(;
    output_type = "--output-exe",
    file = joinpath(@__DIR__, "App"),
    trim_mode = "safe",
    add_ccallables = false,
    verbose = true,
)
link_recipe = JuliaC.LinkRecipe(; image_recipe, outname)
bundle_recipe = JuliaC.BundleRecipe(; link_recipe, output_dir)
JuliaC.compile_products(image_recipe)
JuliaC.link_products(link_recipe)
JuliaC.bundle_products(bundle_recipe)

log = sprint() do io
    cmd = `$(joinpath(output_dir, "bin", "app_test_exe"))`
    return run(pipeline(cmd; stdout = io))
end

println(log)

@test occursin("Ipopt", log)
@test occursin("Violation = ", log)
@test occursin("Optimal Solution Found.", log)
