module Plot

    export Figure, Axis, Axis3
    export hist, scatter, linesegments, mesh,
           hist!, scatter!, linesegments!, mesh!


    using GLMakie: Axis, Axis3, Figure, LineSegments, Mesh, Scatter
    using GLMakie: current_axis, record, hist, hist!
    import GLMakie: linesegments, linesegments!, mesh, mesh!, scatter, scatter!,
                    convert_arguments

    using ..Geometry

    include("point.jl")
    include("linesegment.jl")
end
