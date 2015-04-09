using BinDeps

@BinDeps.setup

windllname = "libipopt-1"
libipopt = library_dependency("libipopt", aliases=[windllname])

ipoptname = "Ipopt-3.12.1"

provides(Sources, URI("http://www.coin-or.org/download/source/Ipopt/$ipoptname.tgz"),
    libipopt, os = :Unix)

prefix=joinpath(BinDeps.depsdir(libipopt),"usr")
patchdir=BinDeps.depsdir(libipopt)
srcdir = joinpath(BinDeps.depsdir(libipopt),"src",ipoptname)

provides(SimpleBuild,
    (@build_steps begin
        GetSources(libipopt)
        @build_steps begin
            ChangeDirectory(srcdir)
            @build_steps begin
                ChangeDirectory(joinpath(srcdir,"ThirdParty","Blas"))
                CreateDirectory("build", true)
                `./get.Blas`
            end
            @build_steps begin
                ChangeDirectory(joinpath(srcdir,"ThirdParty","Blas","build"))
                `../configure --prefix=$prefix --disable-shared --with-pic`
                `make install`
            end
            @build_steps begin
                ChangeDirectory(joinpath(srcdir,"ThirdParty","Lapack"))
                CreateDirectory("build", true)
                `./get.Lapack`
            end
            @build_steps begin
                ChangeDirectory(joinpath(srcdir,"ThirdParty","Lapack","build"))
                `../configure --prefix=$prefix --disable-shared --with-pic`
                `make install`
            end
            @build_steps begin
                ChangeDirectory(joinpath(srcdir,"ThirdParty","ASL"))
                `./get.ASL`
            end
            @build_steps begin
                ChangeDirectory(joinpath(srcdir,"ThirdParty","Mumps"))
                `./get.Mumps`
            end
            `./configure --prefix=$prefix coin_skip_warn_cxxflags=yes
                         --with-blas="$prefix/lib/libcoinblas.a -lgfortran"
                         --with-lapack=$prefix/lib/libcoinlapack.a`
            `make`
            `make test`
            `make -j1 install`
        end
    end),libipopt, os = :Unix)

# OS X
@osx_only begin
    using Homebrew
    provides(Homebrew.HB, "ipopt", libipopt, os = :Darwin)
end


# Windows
@windows_only begin
    using WinRPM
    provides(WinRPM.RPM, "Ipopt", [libipopt], os = :Windows)
end

@BinDeps.install [:libipopt => :libipopt]
