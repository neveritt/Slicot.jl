## Simplified wrapper for SLICOT routines ##

module Simple

import Slicot: SlicotException, BlasInt, libslicot, Raw

export mb02jd, tb04ad




include("simple/mb02jd.jl")
include("simple/mb02jx.jl")

include("simple/tb04ad.jl")

end    # module
