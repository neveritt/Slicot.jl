[SLICOT](http://slicot.org/) wrapper for julialang.

Status
------
Unfinished, untested, extremely new. 

* All functions have been wrapped in the "RAW" API
* So far only 1 routine has been wrapped in the "Simple" API.

However, more are on the way. 

Installation
------------
Git clone the repo into your Julia directory. Change into ./src/slicot,
and run the makefile included there. This will build libslicot.so, and place
it in ./src/slicot/. Move this file somewhere Julia can find it. The 
installation procedure could use some serious work. Research into BinDeps 
will be done in once the library is more mature.

Use
---
There are two submodules: "simple" and "raw". 

Simple makes assumptions about your use case, but allows for a much 
shorter call signature. Inputs are validated inside Julia (allows for 
helpful error messages). Output only arrays are created in the subroutine,
and array dimensions are interpreted from the input arrays (no need for 
explicit passing).

Raw makes no assumptions and leaves all calls up to you. The inputs are
not sanitized in julia, and will result in a call to xerbla.f in the case
of an error. An alternate version of xerbla.f has been included that
doesn't exit upon error, allowing these errors to be handled more
gracefully. **If using the raw bindings, always check if INFO < 0 after
each call**. In addition to this, a "SlicotError" will be thrown as well.