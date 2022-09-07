using UM2


# Get the areas of each radial region
function get_radial_areas(radii::Vector{Float64}, pitch::Float64)
    areas = Vector{Float64}(undef, length(radii) + 1)
    areas[1] = π * radii[1]^2
    for ir = 2:length(radii)
        areas[ir] = π * (radii[ir]^2 - radii[ir-1]^2)
    end
    areas[end] = pitch^2 - π * radii[end]^2
    return areas
end

# Get all radii and areas after splitting each radial region with its radial divisions
function get_divided_radii_and_areas(
        radii::Vector{Float64}, 
        radial_divisions::Vector{Int64},
        radial_areas::Vector{Float64},
        pitch::Float64)
    div_areas = Vector{Float64}(undef, sum(radial_divisions) + 1)
    div_radii = Vector{Float64}(undef, sum(radial_divisions))
    # Inside the innermost ring 
    div_areas[1] = radial_areas[1] / radial_divisions[1]
    div_radii[1] = sqrt(div_areas[1] / π)
    # All other rings
    for ir in 2:radial_divisions[1]
        div_areas[ir] = radial_areas[1] / radial_divisions[1]
        div_radii[ir] = sqrt(div_areas[ir] / π + div_radii[ir-1]^2)
    end
    ctr = radial_divisions[1] + 1
    for ir in 2:length(radii)
        nrad = radial_divisions[ir]
        for _ in 1:nrad
            div_areas[ctr] = radial_areas[ir] / nrad
            div_radii[ctr] = sqrt(div_areas[ctr] / π + div_radii[ctr-1]^2)
            ctr += 1
        end
    end
    # Outside the outermost ring
    div_areas[end] = pitch^2 - π * div_radii[end]^2
    return div_radii, div_areas
end

# Get equivalent radii for the quadrilaterals to preserve areas
function get_equiv_quad_radii(
        div_radii::Vector{Float64}, 
        div_areas::Vector{Float64}, 
        pitch::Float64,
        azimuthal_divisions::Int64)
    θ = 2 * π / azimuthal_divisions
    equiv_radii = Vector{Float64}(undef, length(div_radii))
    # The innermost radius is a special case, and is essentially a triangle.
    # A_t = l² * sin(θ) / 2
    equiv_radii[1] = sqrt(2 * div_areas[1] / (sin(θ) * azimuthal_divisions))
    # A_q = (l² - l²₀) * sin(θ) / 2
    for ir in 2:length(div_radii)
        equiv_radii[ir] = sqrt(2 * div_areas[ir] / (sin(θ) * azimuthal_divisions) + equiv_radii[ir-1]^2)
    end
    if any(r -> r > pitch/2, equiv_radii)
        error("Equivalent radii are greater than half the pitch")
    end

    # Sanity check via area error comparision 
    Ap = Vector{Float64}(undef, length(div_radii))
    Ap[1] = equiv_radii[1]^2 * sin(θ) / 2
    for ir in 2:length(equiv_radii)
        Ap[ir] = (equiv_radii[ir]^2 - equiv_radii[ir-1]^2) * sin(θ) / 2
    end
    if any(i -> abs(div_areas[i] - azimuthal_divisions * Ap[i]) > 1e-6, 1:length(Ap))
        error("Circular and polygon areas are not equal")
    end

    return equiv_radii
end

# Generate the mesh points 
function get_quad_mesh_points(
        equiv_radii::Vector{Float64}, 
        pitch::Float64, 
        azimuthal_divisions::Int64)
    θ = 2 * π / azimuthal_divisions
    points = Point2d[]
    push!(points, Point2d(0.0, 0.0))
    # Inside the innermost ring
    for ia in 1:2:azimuthal_divisions
        r = equiv_radii[1] / 2
        push!(points, Point2d(r * cos(ia * θ), 
                              r * sin(ia * θ)))
    end
    # All other rings
    for ir in 1:length(div_radii)
        for ia in 0:azimuthal_divisions - 1
            push!(points, Point2d(equiv_radii[ir] * cos(ia * θ), 
                                  equiv_radii[ir] * sin(ia * θ)))
        end
    end
    # Outside the outermost ring
    for ia in 0:azimuthal_divisions - 1
        if abs(cos(ia * θ)) < 1e-3 # Avoid singularity/numerical error
            r = pitch / 2
        else
            r = abs(pitch / (2 * cos(ia * θ)))
        end
        push!(points, Point2d(r * cos(ia * θ), 
                              r * sin(ia * θ)))
    end

    return points
end

# Generate the mesh faces
function get_quad_mesh_faces(
        total_radial_divisions::Int64,
        azimuthal_divisions::Int64)
    # Error checking for rings outside the pitch should have been done previously,
    # so we skip that here.
    faces = NTuple{4, Int64}[]
    # Inside the innermost ring
    na = azimuthal_divisions
    na2 = na ÷ 2
    for ia = 1:2:na
        p1 = 1
        p2 = ia + (1 + na2)
        p3 = ia + 1 + (1 + na2)
        p4 = ia ÷ 2 + 2
        p5 = ia + 2 + (1 + na2)
        if p5 == 1 + na + (1 + na2)
            p5 = 1 + (1 + na2)
        end
        push!(faces, (p1, p2, p3, p4))
        push!(faces, (p1, p4, p3, p5))  
    end
    # Rings
    nr = total_radial_divisions - 1
    for ir = 1:nr
        for ia = 1:na
            p1 = ia     + (ir - 1) * na + (1 + na2) 
            p2 = ia     + (ir    ) * na + (1 + na2) 
            p3 = ia + 1 + (ir    ) * na + (1 + na2) 
            p4 = ia + 1 + (ir - 1) * na + (1 + na2) 
            if p3 == 1 + (ir + 1) * na + (1 + na2)
                p3 -= na
            end
            if p4 == 1 + (ir    ) * na + (1 + na2)
                p4 -= na
            end
            push!(faces, (p1, p2, p3, p4))
        end
    end
    # Outside the outermost ring
    for ia = 1:na
        p1 = ia +     (nr    ) * na + (1 + na2) 
        p2 = ia +     (nr + 1) * na + (1 + na2)
        p3 = ia + 1 + (nr + 1) * na + (1 + na2) 
        p4 = ia + 1 + (nr    ) * na + (1 + na2)
        if p3 == 1 + na * (nr + 2) + (1 + na2)
            p3 -= na
        end
        if p4 == 1 + na * (nr + 1) + (1 + na2)
            p4 -= na
        end
        push!(faces, (p1, p2, p3, p4))
    end

    return faces
end

function write_quad_mesh(
        filename::String,
        points::Vector{Point2d},
        faces::Vector{NTuple{4, Int64}},
        radial_divisions::Vector{Int64},
        azimuthal_divisions::Int64,
        materials::Vector{String})
    # Write the file
    io = open(filename, "w");
    try
        println(io, "*Heading")
        println(io, " " * filename)
        println(io, "*NODE")
        for (i, p) in enumerate(points)
            println(io, i, ", ", p[1], ", ", p[2], ", 0.0")
        end
        println(io, "*ELEMENT, type=CPS4, ELSET=ALL")
        for (i, f) in enumerate(faces)
            println(io, i, ", ", f[1], ", ", f[2], ", ", f[3], ", ", f[4])
        end
        fctr = 1
        for (i, mat) in enumerate(materials)
            if i == 1 || mat != materials[i-1]
                println(io, "*ELSET,ELSET=Material:_" * mat)
                ndiv = radial_divisions[i]
            else
                ndiv = 1
            end
            for j in 1:ndiv
                for k in 1:azimuthal_divisions - 1
                    print(io, fctr, ", ")
                    fctr += 1
                end
                print(io, fctr, ",\n")
                fctr += 1
            end
        end
    catch e
        println(e)
    finally
        close(io)
    end
    return nothing
end
