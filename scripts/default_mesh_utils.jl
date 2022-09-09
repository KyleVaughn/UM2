using UM2

# Get the areas of each radial region
function get_radial_areas(r::Vector{Float64}, pitch::Float64)
    a = Vector{Float64}(undef, length(r) + 1)
    a[1] = π * r[1]^2
    for i = 2:length(r)
        a[i] = π * (r[i]^2 - r[i-1]^2)
    end
    a[end] = pitch^2 - π * r[end]^2
    return a
end

# Get all radii and areas after splitting each radial region with its radial divisions
function get_divided_radii_and_areas(
        r_mat::Vector{Float64}, 
        rdivs::Vector{Int64},
        pitch::Float64)
    a_mat = get_radial_areas(r_mat, pitch)
    r_div = Vector{Float64}(undef, sum(rdivs))
    a_div = Vector{Float64}(undef, sum(rdivs) + 1)
    # Inside the innermost ring 
    a_div[1] = r_mat[1] / rdivs[1]
    r_div[1] = sqrt(a_div[1] / π)
    for i in 2:rdivs[1]
        a_div[i] = a_div[1]
        r_div[i] = sqrt(a_div[i] / π + r_div[i-1]^2)
    end
    # All other rings
    ctr = rdivs[1] + 1
    for i in 2:length(r_mat)
        ndiv = rdivs[i]
        a = a_mat[i] / ndiv
        for _ in 1:ndiv
            a_div[ctr] = a
            r_div[ctr] = sqrt(a / π + r_div[ctr-1]^2)
            ctr += 1
        end
    end
    # Outside the outermost ring
    a_div[end] = pitch^2 - π * r_div[end]^2
    return r_div, a_div
end

# Get equivalent radii for the quadrilaterals to preserve areas
function get_equiv_quad_radii(
        r_div::Vector{Float64}, 
        a_div::Vector{Float64}, 
        pitch::Float64)
    θ = 2 * π / n_azi
    r_equiv = Vector{Float64}(undef, length(r_div))
    # The innermost radius is a special case, and is essentially a triangle.
    # A_t = l² * sin(θ) / 2
    r_equiv[1] = sqrt(2 * a_div[1] / (sin(θ) * n_azi))
    # A_q = (l² - l²₀) * sin(θ) / 2
    for ir in 2:length(r_div)
        r_equiv[ir] = sqrt(2 * a_div[ir] / (sin(θ) * n_azi) + r_equiv[ir-1]^2)
    end
    if any(r -> r > pitch/2, r_equiv)
        error("Equivalent radii are greater than half the pitch")
    end

    # Sanity check via area error comparision 
    Ap = Vector{Float64}(undef, length(r_div))
    Ap[1] = r_equiv[1]^2 * sin(θ) / 2
    for ir in 2:length(r_equiv)
        Ap[ir] = (r_equiv[ir]^2 - r_equiv[ir-1]^2) * sin(θ) / 2
    end
    if any(i -> abs(a_div[i] - n_azi * Ap[i]) > 1e-6, 1:length(Ap))
        error("Circular and polygon areas are not equal")
    end

    return r_equiv
end

# Get equivalent radii for the quadratic quadrilaterals to preserve areas
function get_equiv_quad8_radii(
        r_div::Vector{Float64}, 
        a_div::Vector{Float64}, 
        pitch::Float64)
    θ = 2 * π / n_azi
    γ = θ / 2
    sinγ = sin(γ)
    cosγ = cos(γ)
    r_equiv = Vector{Float64}(undef, length(r_div))
    # The innermost radius is a special case, and is essentially a quadratic triangle.
    # A_t = r² * sin(θ) / 2
    # A_e = (4 / 3) * r * sin(θ/2) * (R - r)
    r_equiv[1] = r_div[1] * (3γ + sinγ * cosγ) / (4 * sinγ)
#    # A_q = (l² - l²₀) * sin(θ) / 2
    for ir in 2:length(r_div)
        r₀ = r_div[ir - 1]
        r₁ = r_div[ir]
        R₀ = r_equiv[ir - 1]
        quad_part = 3 * (γ - sinγ * cosγ) * (r₁^2 - r₀^2) / (4 * sinγ)
        prev_edge = r₀ * (R₀ - r₀ * cosγ)
        r_equiv[ir] = inv(r₁) * (quad_part + prev_edge) + r₁ * cosγ
    end
    if any(r -> r > pitch/2, r_equiv)
        error("Equivalent radii are greater than half the pitch")
    end

    # Sanity check via area error comparision 
    Ap = Vector{Float64}(undef, length(r_div))
    r = r_div[1]
    R = r_equiv[1]
    Ap[1] = r^2 * sin(θ) / 2 + (4 / 3) * r * sinγ * (R - r * cosγ)
    for ir in 2:length(r_equiv)
        r₀ = r_div[ir - 1]
        r₁ = r_div[ir]
        R₀ = r_equiv[ir - 1]
        R₁ = r_equiv[ir]
        Ap[ir] = (r₁^2 - r₀^2) * sin(θ) / 2 + 
            (4 / 3) * r₁ * sinγ * (R₁ - r₁ * cosγ) -
            (4 / 3) * r₀ * sinγ * (R₀ - r₀ * cosγ)
    end
    if any(i -> abs(a_div[i] - n_azi * Ap[i]) > 1e-6, 1:length(Ap))
        error("Circular and polygon areas are not equal")
    end

    return r_equiv
end

# Generate the mesh points 
function get_quad_mesh_points(r_equiv::Vector{Float64}, pitch::Float64)
    θ = 2 * π / n_azi
    points = Point2d[]
    push!(points, Point2d(0.0, 0.0))
    # The split edge points inside the innermost ring
    # (Triangular quads)
    for ia in 1:2:n_azi
        r = r_equiv[1] / 2
        push!(points, Point2d(r * cos(ia * θ), 
                              r * sin(ia * θ)))
    end
    # All other rings
    for ir in 1:length(r_div)
        for ia in 0:n_azi - 1
            push!(points, Point2d(r_equiv[ir] * cos(ia * θ), 
                                  r_equiv[ir] * sin(ia * θ)))
        end
    end
    # Outside the outermost ring, on the pin boundary
    for ia in 0:n_azi - 1
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

# Generate the mesh points 
function get_quad8_mesh_points(r_equiv::Vector{Float64}, pitch::Float64)
    θ = 2 * π / n_azi
    γ = θ / 2
    points = get_quad_mesh_points(r_equiv, pitch)
    # Quadratic points
    # The split edge points inside the innermost ring
    for ia in 1:2:n_azi
        r = r_div[1] / 4
        push!(points, Point2d(r * cos(ia * θ), 
                              r * sin(ia * θ)))
        push!(points, Point2d(3r * cos(ia * θ), 
                              3r * sin(ia * θ)))
    end
    # The non-split edge points inside the innermost ring
    for ia in 0:2:n_azi - 1
        r = r_div[1] / 2
        push!(points, Point2d(r * cos(ia * θ), 
                              r * sin(ia * θ)))
    end
    # All the quadratic points on θ 
    for ir in 2:length(r_div)
        r₀ = r_div[ir - 1]
        r₁ = r_div[ir]
        for ia in 0:n_azi - 1
            push!(points, Point2d((r₁ + r₀) / 2 * cos(ia * θ), 
                                  (r₁ + r₀) / 2 * sin(ia * θ)))
        end
    end
    # All the quadratic points on (2i - 1)γ for i = 1,2,3...
    for ir in 1:length(r_equiv)
        R = r_equiv[ir]
        for ia in 1:2:(2*n_azi) - 1
            push!(points, Point2d(R * cos(ia * γ), 
                                  R * sin(ia * γ)))
        end
    end

    return points
end

# Generate the mesh faces
function get_quad_mesh_faces(ndiv::Int64)
    # Error checking for rings outside the pitch should have been done previously,
    # so we skip that here.
    faces = NTuple{4, Int64}[]
    # Innermost "triangular" quads 
    # 1 is the center point
    # There are na/2 points on the split edges of the triangular quads
    na = n_azi
    ntri = na ÷ 2
    nin = 1 + ntri # Number of points prior to the innermost ring
    for ia = 1:2:na
        p1 = 1            # Center point
        p2 = ia + nin     # First regular ring point
        p3 = ia + nin + 1 # Second regular ring point
        p4 = ia ÷ 2 + 2   # Split edge tri quad point. (2, 3, 4, ..., na/2 + 1)
        p5 = ia + nin + 2 # Third regular ring point
        if p5 == 1 + na + nin # If we're at the end of the ring
            p5 = 1 + nin
        end
        push!(faces, (p1, p2, p3, p4))
        push!(faces, (p1, p4, p3, p5))  
    end
    # Rings
    nr = ndiv 
    for ir = 1:nr - 1
        for ia = 1:na
            p1 = ia     + (ir - 1) * na + nin # Bottom left point
            p2 = ia     + (ir    ) * na + nin # Bottom right point
            p3 = ia + 1 + (ir    ) * na + nin # Top right point
            p4 = ia + 1 + (ir - 1) * na + nin # Top left point
            if p3 == 1 + (ir + 1) * na + nin # If we're at the end of the ring
                p3 -= na
            end
            if p4 == 1 + (ir    ) * na + nin # If we're at the end of the ring
                p4 -= na
            end
            push!(faces, (p1, p2, p3, p4))
        end
    end
    # Outside the outermost ring, on the pin boundary
    for ia = 1:na
        p1 = ia +     (nr - 1) * na + nin 
        p2 = ia +     (nr    ) * na + nin
        p3 = ia + 1 + (nr    ) * na + nin 
        p4 = ia + 1 + (nr - 1) * na + nin
        if p3 == 1 + na * (nr + 1) + nin 
            p3 -= na
        end
        if p4 == 1 + na * (nr    ) + nin 
            p4 -= na
        end
        push!(faces, (p1, p2, p3, p4))
    end

    return faces
end

# Generate the mesh faces
function get_quad8_mesh_faces(ndiv::Int64)
    # Error checking for rings outside the pitch should have been done previously,
    # so we skip that here.
    faces = NTuple{8, Int64}[]
    # 1, 6, 7, 2, 62, 98, 55, 54
    # 1, 2, 7, 8, 54, 55, 99, 63
    # Innermost "triangular" quads 
    # 1 is the center point
    # There are na/2 points on the split edges of the triangular quads
    na = n_azi
    ntri = na ÷ 2 
    ntri2 = na # Number of points splitting the triangular split edges
    nin = 1 + ntri # Number of points prior to the innermost ring
    nr = ndiv
    # Number of points in the linear mesh:
    nlin = (nr + 1) * na + nin
    # Number of points places at each θ for the quadratic mesh, 
    # excluding the split edge
    nθ = (nr - 1) * na 
    for ia = 1:2:na
        p1 = 1            # Center point
        p2 = nin + ia     # First regular ring point
        p3 = nin + ia + 1 # Second regular ring point
        p4 = ia ÷ 2 + 2   # Split edge tri quad point. (2, 3, 4, ..., na/2 + 1)
        p5 = nin + ia  + 2 # Third regular ring point
        if p5 == nin + na + 1 # If we're at the end of the ring
            p5 = nin + 1
        end
        p6 = nlin + ntri2 + ia ÷ 2 + 1     # First regular quadratic ring point
        p7 = nlin + ntri2 + ntri + nθ + ia # First γ angle quadratic ring point
        p8 = nlin + ia + 1                 # Second split edge quadratic ring point
        p9 = nlin + ia                     # First split edge quadratic ring point
        p10 = nlin + ntri2 + ntri + nθ + ia + 1 # Second γ angle quadratic ring point
        p11 = nlin + ntri2 + ia ÷ 2 + 2    # Second regular quadratic ring point
        if p11 == nlin + ntri2 + ntri + 1 # If we're at the end of the ring
            p11 = nlin + ntri2 + 1
        end
        push!(faces, (p1, p2, p3, p4, p6, p7, p8, p9))
        push!(faces, (p1, p4, p3, p5, p9, p8, p10, p11))
    end
#    # Rings
#    nr = total_radial_divisions - 1
#    for ir = 1:nr
#        for ia = 1:na
#            p1 = ia     + (ir - 1) * na + (1 + na2) 
#            p2 = ia     + (ir    ) * na + (1 + na2) 
#            p3 = ia + 1 + (ir    ) * na + (1 + na2) 
#            p4 = ia + 1 + (ir - 1) * na + (1 + na2) 
#            if p3 == 1 + (ir + 1) * na + (1 + na2)
#                p3 -= na
#            end
#            if p4 == 1 + (ir    ) * na + (1 + na2)
#                p4 -= na
#            end
#            push!(faces, (p1, p2, p3, p4))
#        end
#    end
#    # Outside the outermost ring
#    for ia = 1:na
#        p1 = ia +     (nr    ) * na + (1 + na2) 
#        p2 = ia +     (nr + 1) * na + (1 + na2)
#        p3 = ia + 1 + (nr + 1) * na + (1 + na2) 
#        p4 = ia + 1 + (nr    ) * na + (1 + na2)
#        if p3 == 1 + na * (nr + 2) + (1 + na2)
#            p3 -= na
#        end
#        if p4 == 1 + na * (nr + 1) + (1 + na2)
#            p4 -= na
#        end
#        push!(faces, (p1, p2, p3, p4))
#    end

    return faces
end

function write_quad_mesh(
        filename::String,
        points::Vector{Point2d},
        faces::Vector{NTuple{4, Int64}},
        radial_divisions::Vector{Int64},
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
                for k in 1:n_azi - 1
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
