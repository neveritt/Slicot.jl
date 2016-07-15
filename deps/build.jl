using BinDeps

BinDeps.@setup

# Locate the BLAS and LAPACK libraries Julia is using
libblas       = library_dependency(Base.libblas_name)
libblaspath   = Libdl.dlpath(libblas.name)
liblapack     = library_dependency(Base.liblapack_name)
liblapackpath = Libdl.dlpath(liblapack.name)

# Define library dependencies accordingly
libslicot = library_dependency("libslicot", aliases = ["slicot"],
                              runtime = true, depends = [libblas, liblapack])

prefix    = joinpath(BinDeps.depsdir(libslicot), "usr")
builddir  = joinpath(BinDeps.depsdir(libslicot), "builds", "libslicot")
srcdir    = joinpath(BinDeps.depsdir(libslicot), "src", "libslicot")

# Provide the build step
buildstep = @build_steps begin
  CreateDirectory(joinpath(builddir))
  @build_steps begin
    ChangeDirectory(builddir)
    FileRule(joinpath(prefix, "lib", "libslicot."*BinDeps.shlib_ext),
      @build_steps begin
        `cmake -DBlasLibrary=$(libblaspath) -DLapackLibrary=$(liblapackpath) -DCMAKE_INSTALL_PREFIX=$(prefix) $(srcdir)`
        `make install`
      end
     )
  end
end

provides(SimpleBuild, buildstep, libslicot)

# Install the library
BinDeps.@install Dict([(:libslicot, :_jl_libslicot)])
