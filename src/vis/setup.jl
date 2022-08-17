export Figure, Axis, Axis3

export hist!, scatter!, linesegments!, mesh!

using GLMakie: Axis, Axis3, Figure, Scatter, LineSegments
using GLMakie: Mesh as GLMakieMesh
using GLMakie: current_axis, record, hist, hist!

import GLMakie: linesegments!, mesh!, scatter!, convert_arguments

const UM2_VIS_NONLINEAR_SUBDIVISIONS = 4
