export Figure, Axis, Axis3
export hist, scatter, linesegments, mesh,
       hist!, scatter!, linesegments!, mesh!


using GLMakie: Axis, Axis3, Figure, Scatter, LineSegments
using GLMakie: Mesh as GLMakieMesh
using GLMakie: current_axis, record, hist, hist!
import GLMakie: linesegments, linesegments!, mesh, mesh!, scatter, scatter!,
                convert_arguments

const plot_nonlinear_subdivisions = 5

# geometry
include("point.jl")
include("polytope.jl")
include("axisalignedbox.jl")

# mesh
include("rectilinear_grid.jl")
