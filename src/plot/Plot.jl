module Plot

    export Figure, Axis, Axis3
    export hist, scatter, linesegments, mesh,
           hist!, scatter!, linesegments!, mesh!


    using StaticArrays
    using GLMakie: Axis, Axis3, Figure, LineSegments, Mesh, Scatter
    using GLMakie: current_axis, record, hist, hist!
    import GLMakie: linesegments, linesegments!, mesh, mesh!, scatter, scatter!,
                    convert_arguments

    using ..Geometry

    const plot_nonlinear_subdivisions = 5

    include("common.jl")
    include("point.jl")
    include("linesegment.jl")
    include("quadraticsegment.jl")
    include("polytope.jl")
    include("axisalignedbox.jl")
end
