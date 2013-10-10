using BinDeps

@BinDeps.setup

libipopt = library_dependency("libipopt")

ipoptname = "Ipopt-3.11.4"

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
            `cat $patchdir/ipopt-shlibs.patch` |> `patch -p1`
            @build_steps begin
                ChangeDirectory(joinpath(srcdir,"ThirdParty","Mumps"))
                `./get.Mumps`
            end
            `./configure --prefix=$prefix`
            `make install`
        end
    end),libipopt, os = :Unix)

@BinDeps.install
