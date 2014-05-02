using BinDeps

@BinDeps.setup

windllname = "IpOpt-vc10"
libipopt = library_dependency("libipopt", aliases=[windllname])

ipoptname = "Ipopt-3.11.7"

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
                ChangeDirectory(joinpath(srcdir,"ThirdParty","Mumps"))
                `./get.Mumps`
            end
            `./configure --prefix=$prefix --enable-dependency-linking --with-blas=$prefix/lib/libcoinblas.a --with-lapack=$prefix/lib/libcoinlapack.a`
            `make install`
        end
    end),libipopt, os = :Unix)

# OS X
@osx_only begin
    using Homebrew
    provides(Homebrew.HB, "ipopt", libipopt, os = :Darwin)
end


# Windows
downloadsdir = BinDeps.downloadsdir(libipopt)
libdir = BinDeps.libdir(libipopt)
downloadname = "Ipopt-3.11.0-Win32-Win64-dll.7z"
windir = WORD_SIZE == 32 ? "Win32" : "x64"

# BinDeps complains about the .7z file on other platforms...
@windows_only provides(BuildProcess,
    (@build_steps begin
        FileDownloader("http://www.coin-or.org/download/binary/Ipopt/$downloadname", joinpath(downloadsdir, downloadname))
	CreateDirectory(BinDeps.srcdir(libipopt), true)
	FileUnpacker(joinpath(downloadsdir, downloadname), BinDeps.srcdir(libipopt), joinpath(BinDeps.srcdir(libipopt),"lib","Win32"))
	CreateDirectory(libdir, true)
	@build_steps begin
	    ChangeDirectory(joinpath(BinDeps.srcdir(libipopt),"lib",windir,"ReleaseMKL"))
	    FileRule(joinpath(libdir,"$(windllname).dll"), @build_steps begin
	        `cp *.dll $(libdir)`
	    end)
	end
     end), libipopt, os = :Windows)

@windows_only push!(BinDeps.defaults, BuildProcess)

@BinDeps.install [:libipopt => :libipopt]

@windows_only pop!(BinDeps.defaults)
