function classify_NESW(p::Point_2D{T}, 
                       mesh::UnstructuredMesh_2D{T, I}) where {T <: AbstractFloat, I <: Unsigned}
    y_N = mesh.points[mesh.edges[mesh.boundary_edges[1][1]][1]].x[2] 
    x_E = mesh.points[mesh.edges[mesh.boundary_edges[2][1]][1]].x[1]
    y_S = mesh.points[mesh.edges[mesh.boundary_edges[3][1]][1]].x[2]
    x_W = mesh.points[mesh.edges[mesh.boundary_edges[4][1]][1]].x[1]
    if abs(p.x[2] - y_N) < 1e-4
        return 1 # North
    elseif abs(p.x[1] - x_E) < 1e-4
        return 2 # East
    elseif abs(p.x[2] - y_S) < 1e-4
        return 3 # South
    elseif abs(p.x[1] - x_W) < 1e-4
        return 4 # West
    else
        return 0 # Error
    end
end

function get_start_edge_NESW(p::Point_2D{T}, 
                             boundary_edge_indices::Vector{I},
                             NESW::Int64,
                             mesh::UnstructuredMesh_2D{T, I}) where {T <: AbstractFloat, I <: Unsigned}
    

    if 0 < length(mesh.edges_materialized)
        if NESW == 1 || NESW == 3
            # On North or South edge. Just check x coordinates
            xₚ = p.x[1]
            for iedge in boundary_edge_indices
                edge_points = mesh.edges_materialized[iedge].points
                x₁ = edge_points[1].x[1]
                x₂ = edge_points[2].x[1]
                if x₁ ≤ xₚ ≤ x₂ || x₂ ≤ xₚ ≤ x₁
                    return iedge
                end
            end
        else # NESW == 2 || NESW == 4
            # On East or West edge. Just check y coordinates
            yₚ = p.x[2]                                      
            for iedge in boundary_edge_indices
                edge_points = mesh.edges_materialized[iedge].points
                y₁ = edge_points[1].x[2]
                y₂ = edge_points[2].x[2]
                if y₁ ≤ yₚ ≤ y₂ || y₂ ≤ yₚ ≤ y₁
                    return iedge
                end
            end
        end
    else # Not materialized edges
        if NESW == 1 || NESW == 3
            # On North or South edge. Just check x coordinates
            xₚ = p.x[1]
            for iedge in boundary_edge_indices
                edge_points = get_edge_points(mesh, mesh.edges[iedge])
                x₁ = edge_points[1].x[1]
                x₂ = edge_points[2].x[1]
                if x₁ ≤ xₚ ≤ x₂ || x₂ ≤ xₚ ≤ x₁
                    return iedge
                end
            end
        else # NESW == 2 || NESW == 4
            # On East or West edge. Just check y coordinates
            yₚ = p.x[2]                                      
            for iedge in boundary_edge_indices
                edge_points = get_edge_points(mesh, mesh.edges[iedge])
                y₁ = edge_points[1].x[2]
                y₂ = edge_points[2].x[2]
                if y₁ ≤ yₚ ≤ y₂ || y₂ ≤ yₚ ≤ y₁
                    return iedge
                end
            end
        end
    end
end

function ray_trace(tₛ::T,
                   ang_quad::ProductAngularQuadrature{nᵧ, nₚ, T}, 
                   HRPM::HierarchicalRectangularlyPartitionedMesh{T, I}
                   ) where {nᵧ, nₚ, T <: AbstractFloat, I <: Unsigned}
    the_tracks = tracks(tₛ, ang_quad, HRPM)
    segment_points = segmentize(the_tracks, HRPM)
    nlevels = levels(HRPM)
    template_vec = MVector{nlevels, I}(zeros(I, nlevels))
    face_indices = segment_face_indices(segment_points, HRPM, template_vec)
    return segment_points, face_indices
end

function ray_trace(tₛ::T,
                   ang_quad::ProductAngularQuadrature{nᵧ, nₚ, T}, 
                   mesh::UnstructuredMesh_2D{T, I}
                   ) where {nᵧ, nₚ, T <: AbstractFloat, I <: Unsigned}
    the_tracks = tracks(tₛ, ang_quad, mesh)
    # If the mesh has boundary edges, assume you want to use the combined segmentize
    # face index algorithm
    if 0 < length(mesh.boundary_edges)
        segment_points = segmentize(the_tracks, mesh)
        face_indices = segment_face_indices(segment_points, mesh)
        return segment_points, face_indices
    else
        segment_points = segmentize(the_tracks, mesh)
        face_indices = segment_face_indices(segment_points, mesh)
        return segment_points, face_indices
    end
end

# Get the segment points and face indices for a single track using the
# edge-to-edge ray tracing method. Assumes a rectangular boundary
function ray_trace_edge_to_edge(l::LineSegment_2D{T},
                                mesh::UnstructuredMesh_2D{T, I},
                                bb::Quadrilateral_2D{T}
                                ) where {T <: AbstractFloat, I <: Unsigned}
    # Classify line as intersecting NSEW
    bb = AABB(mesh, rectangular_boundary=true)
    npoints, (start_point, end_point) = l ∩ bb
    start_NESW = classify_NESW(start_point, mesh)
    if start_NESW == 0 
        @warn "Could not classify track start point $start_point"
    end
    iedge = get_start_edge_NESW(start_point, mesh.boundary_edges[start_NESW], start_NESW, mesh)
    iface = mesh.edge_face_connectivity[iedge][2] # 1st entry should be 0
    iedge_old = iedge
    println("iface: $iface")
    println("iedge: $iedge")
    intersection_points = [start_point]
    face_indices = [I(iface)]
    end_reached = false
    f = Figure()
    ax = Axis(f[1, 1], aspect = 1)
#    linesegments!(bb)
    linesegments!(l)
    if 0 < length(mesh.edges_materialized)
        while !end_reached
           # For each edge in this face, intersect the track with the edge
            for edge_id in mesh.face_edge_connectivity[iface]             
                if edge_id == iedge_old
                    continue
                end
                linesegments!(mesh.edges_materialized[edge_id])
                npoints, points = l ∩ mesh.edges_materialized[edge_id]
                println("edge_id: $edge_id")
                println("npoints: $npoints")
                # If there was an intersection, move to the next face
                if 0 < npoints
                    scatter!(collect(points[1:npoints]))
                    append!(intersection_points, collect(points[1:npoints]))
                    push!(face_indices, I(iface))
                    println("edge_face: ", mesh.edge_face_connectivity[edge_id][1], " ",
                            mesh.edge_face_connectivity[edge_id][2])
                    if mesh.edge_face_connectivity[edge_id][1] == iface
                        iface = mesh.edge_face_connectivity[edge_id][2]
                    else
                        iface = mesh.edge_face_connectivity[edge_id][1]
                    end
                    println("new iface: $iface")
                    iedge = edge_id
                    println("new iedge: $iedge")
                end
                s = readline()
            end
            iedge_old = iedge
            last_point = last(intersection_points)
            if distance(last_point, end_point) < 1e-5
                end_reached = true
                if last_point != end_point
                    push!(intersection_points, end_point)
                end
            end
        end
    else # implicit

    end
    return intersection_points
end

function segmentize(tracks::Vector{Vector{LineSegment_2D{T}}},
                    HRPM::HierarchicalRectangularlyPartitionedMesh{T, I}
                    ) where {T <: AbstractFloat, I <: Unsigned}

    # Give info about intersection algorithm being used
    int_alg = get_intersection_algorithm(HRPM) 
    @info "Segmentizing using the '$int_alg' algorithm"
    # index 1 = γ
    # index 2 = track
    # index 3 = point/segment
    seg_points = Vector{Vector{Vector{Point_2D{T}}}}(undef, length(tracks))
    Threads.@threads for iγ = 1:length(tracks)
        # for each track, intersect the track with the mesh
        seg_points[iγ] = tracks[iγ] .∩ HRPM
    end
    return seg_points
end

function segmentize(tracks::Vector{Vector{LineSegment_2D{T}}},
                    mesh::UnstructuredMesh_2D{T, I}
                    ) where {T <: AbstractFloat, I <: Unsigned}
    # Give info about intersection algorithm being used
    int_alg = get_intersection_algorithm(mesh) 
    @info "Segmentizing using the '$int_alg' algorithm"
    # index 1 = γ
    # index 2 = track
    # index 3 = point/segment
    seg_points = Vector{Vector{Vector{Point_2D{T}}}}(undef, length(tracks))
    Threads.@threads for iγ = 1:length(tracks)
        # for each track, intersect the track with the mesh
        seg_points[iγ] = tracks[iγ] .∩ mesh
    end
    return seg_points
end

function segment_face_indices(seg_points::Vector{Vector{Vector{Point_2D{T}}}},
                              HRPM::HierarchicalRectangularlyPartitionedMesh{T, I},
                              template_vec::MVector{N, I}
                             ) where {T <: AbstractFloat, I <: Unsigned, N}

    @info "Finding face indices corresponding to each segment"
    if !are_faces_materialized(HRPM)
        @warn "Faces are not materialized for this mesh. This will be VERY slow"
    end
    nγ = length(seg_points)
    bools = fill(false, nγ)
    # Preallocate indices in the most frustrating way
    indices =   [    
                    [ 
                        [ 
                            MVector{N, I}(zeros(I, N)) 
                                for i = 1:length(seg_points[iγ][it])-1 # Segments
                        ] for it = 1:length(seg_points[iγ]) # Tracks
                    ] for iγ = 1:nγ # Angles
                ]
    Threads.@threads for iγ = 1:nγ
        bools[iγ] = segment_face_indices!(iγ, seg_points[iγ], indices[iγ], HRPM)
    end
    if !all(bools)
        it_bad = findall(x->!x, bools)
        @warn "Failed to find indices for some points in seg_points[$it_bad]"
    end
    return indices
end

# Get the face indices for all tracks in a single angle
function segment_face_indices!(iγ::Int64,
                              points::Vector{Vector{Point_2D{T}}},
                              indices::Vector{Vector{MVector{N, I}}},
                              HRPM::HierarchicalRectangularlyPartitionedMesh{T, I}
                             ) where {T <: AbstractFloat, I <: Unsigned, N}
    nt = length(points)
    bools = fill(false, nt)
    # for each track, find the segment indices
    for it = 1:nt
        # Points in the track
        npoints = length(points[it])
        # Returns true if indices were found for all segments in the track
        bools[it] = segment_face_indices!(points[it], indices[it], HRPM)
    end
    return all(bools)
end

# Get the face indices for all segments in a single track
function segment_face_indices!(points::Vector{Point_2D{T}},
                              indices::Vector{MVector{N, I}},
                              HRPM::HierarchicalRectangularlyPartitionedMesh{T, I}
                             ) where {T <: AbstractFloat, I <: Unsigned, N}
    # Points in the track
    npoints = length(points)
    bools = fill(false, npoints-1)
    # Test the midpoint of each segment to find the face
    for iseg = 1:npoints-1
        p_midpoint = midpoint(points[iseg], points[iseg+1])
        bools[iseg] = find_face(p_midpoint, indices[iseg], HRPM)
    end
    return all(bools)
end

function segment_face_indices(seg_points::Vector{Vector{Vector{Point_2D{T}}}},
                              mesh::UnstructuredMesh_2D{T, I}
                             ) where {T <: AbstractFloat, I <: Unsigned, N}

    @info "Finding face indices corresponding to each segment"
    if !(0 < length(mesh.faces_materialized))
        @warn "Faces are not materialized for this mesh. This will be VERY slow"
    end
    nγ = length(seg_points)
    bools = fill(false, nγ)
    # Preallocate indices in the most frustrating way
    indices =   [    
                    [ 
                        [ 
                            I(0) 
                                for i = 1:length(seg_points[iγ][it])-1 # Segments
                        ] for it = 1:length(seg_points[iγ]) # Tracks
                    ] for iγ = 1:nγ # Angles
                ]
    Threads.@threads for iγ = 1:nγ
        bools[iγ] = segment_face_indices!(iγ, seg_points[iγ], indices[iγ], mesh)
    end
    if !all(bools)
        it_bad = findall(x->!x, bools)
        @warn "Failed to find indices for some points in seg_points[$it_bad]"
    end
    return indices
end

# Get the face indices for all tracks in a single angle
function segment_face_indices!(iγ::Int64,
                               points::Vector{Vector{Point_2D{T}}},
                               indices::Vector{Vector{I}},
                               mesh::UnstructuredMesh_2D{T, I}
                              ) where {T <: AbstractFloat, I <: Unsigned, N}
    nt = length(points)
    bools = fill(false, nt)
    # for each track, find the segment indices
    for it = 1:nt
        # Points in the track
        npoints = length(points[it])
        # Returns true if indices were found for all segments in the track
        bools[it] = segment_face_indices!(points[it], indices[it], mesh)
    end
    return all(bools)
end

# Get the face indices for all segments in a single track
function segment_face_indices!(points::Vector{Point_2D{T}},
                               indices::Vector{I},
                               mesh::UnstructuredMesh_2D{T, I}
                              ) where {T <: AbstractFloat, I <: Unsigned, N}
    # Points in the track
    npoints = length(points)
    bools = fill(false, npoints-1)
    # Test the midpoint of each segment to find the face
    for iseg = 1:npoints-1
        p_midpoint = midpoint(points[iseg], points[iseg+1])
        indices[iseg] = find_face(p_midpoint, mesh)
        bools[iseg] = 0 < indices[iseg] 
    end
    return all(bools)
end

# Follows https://mit-crpg.github.io/OpenMOC/methods/track_generation.html
function tracks(tₛ::T,
                ang_quad::ProductAngularQuadrature{nᵧ, nₚ, T},
                HRPM::HierarchicalRectangularlyPartitionedMesh{T, I}
                ) where {nᵧ, nₚ, T <: AbstractFloat, I <: Unsigned}
    w = width(HRPM) 
    h = height(HRPM)
    # The tracks for each γ
    the_tracks = [ tracks(tₛ, w, h, γ) for γ in ang_quad.γ ]  
    # Shift all tracks if necessary, since the tracks are generated as if the HRPM has a 
    # bottom left corner at (0,0)
    offset = HRPM.rect.points[1]
    for angle in the_tracks
        for track in angle
            track = LineSegment_2D(track.points[1] + offset, track.points[2] + offset)
        end
    end
    return the_tracks
end

function tracks(tₛ::T,
                ang_quad::ProductAngularQuadrature{nᵧ, nₚ, T},
                mesh::UnstructuredMesh_2D{T, I};
                boundary_shape="Rectangle"
                ) where {nᵧ, nₚ, T <: AbstractFloat, I <: Unsigned}

    if boundary_shape == "Rectangle"
        bb = AABB(mesh, rectangular_boundary=true)
        w = bb.points[3].x[1] - bb.points[1].x[1]
        h = bb.points[3].x[2] - bb.points[1].x[2]
        # The tracks for each γ
        the_tracks = [ tracks(tₛ, w, h, γ) for γ in ang_quad.γ ]  
        # Shift all tracks if necessary, since the tracks are generated as if the HRPM has a 
        # bottom left corner at (0,0)
        offset = bb.points[1]
        for angle in the_tracks
            for track in angle
                track = LineSegment_2D(track.points[1] + offset, track.points[2] + offset)
            end
        end
        return the_tracks
    else
        @error "Unsupported boundary shape"
    end
end

function tracks(tₛ::T, w::T, h::T, γ::T) where {T <: AbstractFloat}
    # Number of tracks in y direction
    n_y = ceil(Int64, w*abs(sin(γ))/tₛ)
    # Number of tracks in x direction
    n_x = ceil(Int64, h*abs(cos(γ))/tₛ)  
    # Total number of tracks
    nₜ = n_y + n_x
    # Allocate the tracks
    the_tracks = Vector{LineSegment_2D{T}}(undef, nₜ)
    # Effective angle to ensure cyclic tracks
    γₑ = atan((h*n_x)/(w*n_y))
    if π/2 < γ
        γₑ = γₑ + T(π/2)
    end
    # Effective ray spacing for the cyclic tracks
    t_eff = w*sin(atan((h*n_x)/(w*n_y)))/n_x
    if γₑ ≤ π/2
        # Generate tracks from the bottom edge of the rectangular domain
        for ix = 1:n_x
            x₀ = w - t_eff*T(ix - 0.5)/sin(γₑ)
            y₀ = T(0)
            # Segment either terminates at the right edge of the rectangle
            # Or on the top edge of the rectangle
            x₁ = min(w, h/tan(γₑ) + x₀)
            y₁ = min((w - x₀)*tan(γₑ), h) 
            l = LineSegment_2D(Point_2D(x₀, y₀), Point_2D(x₁, y₁))
            if arc_length(l) < minimum_segment_length
                @warn "Small track generated: $l"
            end
            the_tracks[ix] = l
        end
        # Generate tracks from the left edge of the rectangular domain
        for iy = 1:n_y
            x₀ = T(0) 
            y₀ = t_eff*T(iy - 0.5)/cos(γₑ)
            # Segment either terminates at the right edge of the rectangle
            # Or on the top edge of the rectangle
            x₁ = min(w, (h - y₀)/tan(γₑ))
            y₁ = min(w*tan(γₑ) + y₀, h)
            l = LineSegment_2D(Point_2D(x₀, y₀), Point_2D(x₁, y₁))
            if arc_length(l) < minimum_segment_length
                @warn "Small track generated: $l"
            end
            the_tracks[n_x + iy] = l
        end
    else
        # Generate tracks from the bottom edge of the rectangular domain
        for ix = n_y:-1:1
            x₀ = w - t_eff*T(ix - 0.5)/sin(γₑ)
            y₀ = T(0)
            # Segment either terminates at the left edge of the rectangle
            # Or on the top edge of the rectangle
            x₁ = max(0, h/tan(γₑ) + x₀)
            y₁ = min(x₀*abs(tan(γₑ)), h) 
            l = LineSegment_2D(Point_2D(x₀, y₀), Point_2D(x₁, y₁))
            if arc_length(l) < minimum_segment_length
                @warn "Small track generated: $l"
            end
            the_tracks[ix] = l
        end
        # Generate tracks from the right edge of the rectangular domain
        for iy = 1:n_x
            x₀ = w
            y₀ = t_eff*T(iy - 0.5)/abs(cos(γₑ))
            # Segment either terminates at the left edge of the rectangle
            # Or on the top edge of the rectangle
            x₁ = max(0, w + (h - y₀)/tan(γₑ))
            y₁ = min(w*abs(tan(γₑ)) + y₀, h)
            l = LineSegment_2D(Point_2D(x₀, y₀), Point_2D(x₁, y₁))
            if arc_length(l) < minimum_segment_length
                @warn "Small track generated: $l"
            end
            the_tracks[n_y + iy] = l
        end
    end
    return the_tracks
end
