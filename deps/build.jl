using BinaryProvider

# Parse some basic command-line arguments
const verbose = "--verbose" in ARGS
const prefix = Prefix(get([a for a in ARGS if a != "--verbose"], 1, joinpath(@__DIR__, "usr")))

# Instantiate products:
libipopt = LibraryProduct(prefix, "libipopt")
libcoinmumps = LibraryProduct(prefix, "libcoinmumps")
products = [libipopt, libcoinmumps]

# Download binaries from hosted location
bin_prefix = "https://github.com/staticfloat/IpoptBuilder/releases/download/v3.12.8-5"

# Listing of files generated by BinaryBuilder:
download_info = Dict(
  BinaryProvider.Linux(:x86_64, :glibc)      => ("$bin_prefix/Ipopt.x86_64-linux-gnu.tar.gz",      "243b8bc378d2899f4cdc4167a2fea90706e5da986aa0f01761b17ad886bdd817"),
  BinaryProvider.Linux(:i686, :glibc)        => ("$bin_prefix/Ipopt.i686-linux-gnu.tar.gz",        "7723342d77ea9b259fce9123f8391d19988172e6809fa820d6b5e1432fe10f67"),
  BinaryProvider.Linux(:aarch64, :glibc)     => ("$bin_prefix/Ipopt.aarch64-linux-gnu.tar.gz",     "79a6c924168fc8fb5918825776c8e2fc4569e8a515135dfc229c281a1d116128"),
  BinaryProvider.Linux(:armv7l, :glibc)      => ("$bin_prefix/Ipopt.arm-linux-gnueabihf.tar.gz",   "7b6962275f5af4d10a699673bc0576dfc222223dbaf10913ea69ed69bc38b24c"),
  BinaryProvider.MacOS()                     => ("$bin_prefix/Ipopt.x86_64-apple-darwin14.tar.gz", "bf608fa62fdc2626c8fbdf2d08302ccf053c15087b4d8f5047456ccad357d588"),
  BinaryProvider.Windows(:x86_64)            => ("$bin_prefix/Ipopt.x86_64-w64-mingw32.tar.gz",    "6e5b034fb85325a823a9e042f5849cbc38d366df30f5e913819b57bb21f0e16e"),
  BinaryProvider.Windows(:i686)              => ("$bin_prefix/Ipopt.i686-w64-mingw32.tar.gz",      "c42038784d9fbc5561badd7ad99b7ec085f6980be4b1539ef19b5193f37c5797"),
)
if platform_key() in keys(download_info)
    # First, check to see if we're all satisfied
    if any(!satisfied(p; verbose=verbose) for p in products)
        # Download and install binaries
        url, tarball_hash = download_info[platform_key()]
        install(url, tarball_hash; prefix=prefix, force=true, verbose=true)
    end

    # Finally, write out a deps.jl file that will contain mappings for each
    # named product here:
    @write_deps_file libipopt libcoinmumps
else
    error("Your platform $(Sys.MACHINE) is not supported by this package!")
end

