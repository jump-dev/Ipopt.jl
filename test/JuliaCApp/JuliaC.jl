# Copyright (c) 2013: Iain Dunning, Miles Lubin, and contributors
#
# Use of this source code is governed by an MIT-style license that can be found
# in the LICENSE.md file or at https://opensource.org/licenses/MIT.

import JuliaC

img = JuliaC.ImageRecipe(;
    output_type = "--output-exe",
    file = joinpath(@__DIR__, "App"),
    trim_mode = "no",
    add_ccallables = false,
    verbose = true,
)
link = JuliaC.LinkRecipe(; image_recipe = img, outname = "build/app_test_exe")
bun = JuliaC.BundleRecipe(; link_recipe = link, output_dir = "build")
JuliaC.compile_products(img)
JuliaC.link_products(link)
JuliaC.bundle_products(bun)
