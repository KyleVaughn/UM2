# For use with creating meshes by hand

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

# Get all radii and areas after splitting each radial region 
# with its radial divisions
function get_divided_radii_and_areas(
        r_mat::Vector{Float64}, 
        rdivs::Vector{Int64},
        pitch::Float64)
    a_mat = get_radial_areas(r_mat, pitch)
    r_div = Vector{Float64}(undef, sum(rdivs))
    a_div = Vector{Float64}(undef, sum(rdivs) + 1)
    # Inside the innermost material 
    a_div[1] = a_mat[1] / rdivs[1]
    r_div[1] = sqrt(a_div[1] / π)
    for i in 2:rdivs[1]
        a_div[i] = a_div[1]
        r_div[i] = sqrt(a_div[i] / π + r_div[i-1]^2)
    end
    # All other materials
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
        n_azi::Int64,
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
        println(r_equiv)
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
        n_azi::Int64,
        pitch::Float64)
    θ = 2 * π / n_azi
    γ = θ / 2
    sinγ = sin(γ)
    cosγ = cos(γ)
    r_equiv = Vector{Float64}(undef, length(r_div))
    # The innermost radius is a special case, and is essentially a quadratic triangle.
    # A_t = r² * sin(θ) / 2
    # A_edge = (4 / 3) * r * sin(θ/2) * (R - r)
    # A_q = (l² - l²₀) * sin(θ) / 2
    r_equiv[1] = r_div[1] * (3γ + sinγ * cosγ) / (4 * sinγ)
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
function get_quad_mesh_points(r_equiv::Vector{Float64}, n_azi::Int64, pitch::Float64)
    # The azimuthal angle
    θ = 2 * π / n_azi
    nrings = length(r_equiv)
    # The number of points is:
    # center point
    # n_azi * (nrings + 1 boundary) 
    # n_azi / 2 points for the triangular quads
    points = Vector{NTuple{2, Float64}}(undef, 1 + n_azi * (nrings + 1) + n_azi ÷ 2)
    # Center point
    points[1] = (0.0, 0.0)
    # The ring + boundary points.
    # Excludes the center point and the points used to convert the inner
    # triangles to quads.
    for ia in 0:n_azi - 1
        nprev_points = 1 + ia * (nrings + 1)
        sinθ, cosθ = sincos(θ * ia)
        if abs(sinθ) < 1e-6
            sinθ = 0.0
        end
        if abs(cosθ) < 1e-6
            cosθ = 0.0
        end
        # Ring points
        for (ir, r) in enumerate(r_equiv)
            points[nprev_points + ir] = (r * cosθ, r * sinθ)
        end
        # Boundary points
        rx = abs(pitch / (2 * cosθ))
        ry = abs(pitch / (2 * sinθ))
        rb = min(rx, ry)
        points[nprev_points + nrings + 1] = (rb * cosθ, rb * sinθ)
    end

    # The split edge points inside the innermost ring
    # (Triangular quads)
    rt = r_equiv[1] / 2
    nprev_points = 1 + n_azi * (nrings + 1)
    for ia in 1:(n_azi ÷ 2)
        sinθ, cosθ = sincos(θ * (2 * ia - 1))
        points[nprev_points + ia] = (rt * cosθ, rt * sinθ)
    end

    return points
end

# Generate the mesh points 
function get_quad8_mesh_points(r_equiv::Vector{Float64}, n_azi::Int64, pitch::Float64)
    θ = 2 * π / n_azi
    γ = θ / 2
    nrings = length(r_equiv)
    # The number of points is:
    # center point
    # 2 * n_azi * (nrings + 1 boundary) 
    # n_azi points for the quadratic points on the triangular quads
    points = Vector{NTuple{2, Float64}}(undef, 1 + 3 * n_azi * (nrings + 1) + n_azi)
    points[1] = (0.0, 0.0)
    # The ring + boundary points.
    # Excludes the center point and the quadratic points on the shared edge.
    for ia in 0:n_azi - 1
        nprev_points = 1 + 3 * ia * (nrings + 1)
        sinθ, cosθ = sincos(θ * ia)
        if abs(sinθ) < 1e-6
            sinθ = 0.0
        end
        if abs(cosθ) < 1e-6
            cosθ = 0.0
        end
        # Ring points
        for ir in 1:nrings
            if ir == 1
                r₀ = 0.0
            else
                r₀ = r_equiv[ir - 1]
            end
            r₁ = r_equiv[ir]
            rq = (r₀ + r₁) / 2
            points[nprev_points + 2 * ir - 1] = (rq * cosθ, rq * sinθ)
            points[nprev_points + 2 * ir    ] = (r₁ * cosθ, r₁ * sinθ)
        end
        # Boundary points
        rx = abs(pitch / (2 * cosθ))
        ry = abs(pitch / (2 * sinθ))
        rb = min(rx, ry)
        rq = (r_equiv[end] + rb) / 2
        points[nprev_points + 2 * nrings + 1] = (rq * cosθ, rq * sinθ)
        points[nprev_points + 2 * nrings + 2] = (rb * cosθ, rb * sinθ)
        # Middle quadratic points
        # Ring points
        sinγ, cosγ = sincos((2 * ia + 1) * γ)
        for ir in 1:nrings
            r = r_equiv[ir]
            points[nprev_points + 2 * (nrings + 1) + ir] = (r * cosγ, r * sinγ)
        end
        # Boundary points
        rx = abs(pitch / (2 * cosγ))
        ry = abs(pitch / (2 * sinγ))
        rb = min(rx, ry)
        points[nprev_points + 3 * nrings + 3] = (rb * cosγ, rb * sinγ)
    end

    # The split edge points inside the innermost ring
    # (Triangular quads)
    r = r_equiv[1]
    r1 = r / 4
    r2 = 3 * r / 4
    nprev_points = 1 + 3 * n_azi * (nrings + 1) 
    for ia in 1:(n_azi ÷ 2)
        sinθ, cosθ = sincos(θ * (2 * ia - 1))
        points[nprev_points + (2 * ia - 1)] = (r1 * cosθ, r1 * sinθ)
        points[nprev_points + (2 * ia    )] = (r2 * cosθ, r2 * sinθ)
    end

    return points
end

# Generate the mesh faces
function get_quad_mesh_faces(ndiv::Int64, n_azi::Int64)
    # Error checking for rings outside the pitch should have been done previously,
    # so we skip that here.
    nfaces = (ndiv + 1) * n_azi
    faces = Vector{NTuple{4, Int64}}(undef, nfaces)
    # Innermost "triangular" quads 
    # 1 is the center point
    na = n_azi
    nrad = ndiv + 1
    nreg = nrad * na + 1
    npoints = nreg + na ÷ 2
    for ia = 1:(na ÷ 2)
        p1 = 1                        # Center point
        p2 = 2 + nrad * 2 * (ia - 1)  # First regular ring point
        p3 = 2 + nrad * 2 * (ia - 1) + nrad   # Second regular ring point
        p4 = nreg + ia                # Split edge tri quad point
        p5 = 2 + nrad * 2 * ia        # Third regular ring point
        # If we're at the end of the ring
        if ia == na ÷ 2
            p5 = 2
        end
        faces[(2 * nrad * (ia - 1)) + 1] = (p1, p2, p3, p4)
        faces[(2 * nrad * (ia - 1)) + 1 + nrad] = (p1, p4, p3, p5)
    end
    # All the other quads
    for ia = 1:na
        for ir = 2:nrad
            p1 = nrad * (ia - 1) + ir
            p2 = nrad * (ia - 1) + ir + 1
            p3 = nrad * (ia - 1) + ir + nrad + 1
            p4 = nrad * (ia - 1) + ir + nrad
            if ia == na
                p3 -= (nreg - 1)
                p4 -= (nreg - 1)
            end
            faces[(ia - 1) * nrad + ir] = (p1, p2, p3, p4)
        end
    end
    return faces
end

# Generate the mesh faces
function get_quad8_mesh_faces(ndiv::Int64, n_azi::Int64)
    # Error checking for rings outside the pitch should have been done previously,
    # so we skip that here.
    nfaces = (ndiv + 1) * n_azi
    faces = Vector{NTuple{8, Int64}}(undef, nfaces)
    na = n_azi
    nrad = ndiv + 1
    nθ = 2 * nrad
    nγ = nrad
    nreg = 1 + 3 * n_azi * nrad
    # Innermost "triangular" quads 
    # 1 is the center point
    for ia = 1:(na ÷ 2)
        p1 = 1                        # Center point
        p5 = 1 + 2 * (nθ + nγ) * (ia - 1) + 1
        p2 = p5 + 1
        p6 = p5 + nθ
        p4 = p6 + nγ
        p3 = p4 + 1
        p8 = nreg + 2 * ia - 1
        p7 = nreg + 2 * ia
        p9 = p4 + nθ
        p10 = p9 + nγ
        p11 = p10 + 1
        # If we're at the end of the ring
        if ia == na ÷ 2
            p10 = 2
            p11 = 3
        end
        faces[(2 * nrad * (ia - 1)) + 1] = (p1, p2, p3, p4, p5, p6, p7, p8)
        faces[(2 * nrad * (ia - 1)) + 1 + nrad] = (p1, p4, p3, p11, p8, p7, p9, p10)
    end

    # All the other quads
    for ia = 1:na
        for ir = 2:nrad
            p1 = 2 * ir - 1 + (nθ + nγ) * (ia - 1)
            p2 = p1 + 2
            p5 = p1 + 1

            p4 = p1 + nγ + nθ
            p3 = p4 + 2
            p7 = p4 + 1 

            p8 = nθ + ir + (nθ + nγ) * (ia - 1)
            p6 = p8 + 1
            if ia == na
                p4 = 1 + 2 * (ir - 1)
                p7 = 2 + 2 * (ir - 1)
                p3 = 3 + 2 * (ir - 1)
            end
            faces[(ia - 1) * nrad + ir] = (p1, p2, p3, p4, p5, p6, p7, p8)
        end
    end

    return faces
end

#function write_quad_mesh(
#        filename::String,
#        pitch::Float64,
#        points::Vector{NTuple{2, Float64}},
#        faces::Vector{NTuple{4, Int64}},
#        rdivs::Vector{Int64},
#        materials::Vector{String},
#        elsets::Vector{String}
#    )
#    # Write the file
#    io = open(filename, "w");
#    try
#        println(io, "*Heading")
#        println(io, " " * filename)
#        println(io, "*NODE")
#        p2 = pitch / 2 # offset to make all points positive
#        for (i, p) in enumerate(points)
#            println(io, i, ", ", p[1] + p2, ", ", p[2] + p2, ", 0.0")
#        end
#        println(io, "*ELEMENT, type=CPS4, ELSET=ALL")
#        for (i, f) in enumerate(faces)
#            println(io, i, ", ", f[1], ", ", f[2], ", ", f[3], ", ", f[4])
#        end
#        fctr = 1
#        for (i, mat) in enumerate(materials)
#            if i == 1 || mat != materials[i-1]
#                println(io, "*ELSET,ELSET=Material:_" * mat)
#                ndiv = rdivs[i]
#            else
#                ndiv = 1
#            end
#            for j in 1:ndiv
#                for k in 1:n_azi - 1
#                    print(io, fctr, ", ")
#                    fctr += 1
#                end
#                print(io, fctr, ",\n")
#                fctr += 1
#            end
#        end
#        nfaces = length(faces)
#        for elset in elsets
#            println(io, "*ELSET,ELSET=" * elset)
#            for i in 1:nfaces
#                if i == nfaces
#                    println(io, i)
#                elseif i % 10 == 0
#                    print(io, i, ",\n")
#                else
#                    print(io, i, ", ")
#                end
#            end
#        end
#    catch e
#        println(e)
#    finally
#        close(io)
#    end
#    return nothing
#end

function write_quad8_mesh(
        filename::String,
        pitch::Float64,
        points::Vector{NTuple{2, Float64}},
        faces::Vector{NTuple{8, Int64}},
        rdivs::Vector{Int64},
        materials::Vector{String},
        elsets::Vector{String}
    )
    # Write the file
    io = open(filename, "w");
    try
        nfaces = length(faces)
        println(io, "*Heading")
        println(io, " " * filename)
        println(io, "*NODE")
        p2 = pitch / 2 # offset to make all points positive
        for (i, p) in enumerate(points)
            println(io, i, ", ", p[1] + p2, ", ", p[2] + p2, ", 0.0")
        end
        println(io, "*ELEMENT, type=CPS8, ELSET=ALL")
        for (i, f) in enumerate(faces)
            println(io, i, ", ", f[1], ", ", f[2], ", ", f[3], ", ", f[4],
                    ", ", f[5], ", ", f[6], ", ", f[7], ", ", f[8])
        end
        material_dict = Dict{String, Set{Int64}}()
        for mat in materials
            if !haskey(material_dict, mat)
                material_dict[mat] = Set{Int64}()
            end
        end
        nrad = sum(rdivs) + 1
        cum_divs = 0
        for (i, mat) in enumerate(materials)
            if i != 1
                cum_divs += rdivs[i-1]
            end
            for ia in 1:n_azi
                if i == length(materials)
                    ndiv = 1
                else
                    ndiv = rdivs[i]
                end
                for ir in 1:ndiv
                    fctr = (ia - 1) * nrad  + ir + cum_divs
                    push!(material_dict[mat], fctr)
                end
            end
        end
        for mat in sort!(collect(keys(material_dict)))
            println(io, "*ELSET,ELSET=Material:_" * mat)
            ids = sort!(collect(material_dict[mat]))
            for (i, id) in enumerate(ids)
                if i == length(ids)
                    println(io, id)
                elseif i % 10 == 0
                    print(io, id, ",\n")
                else
                    print(io, id, ", ")
                end
            end
        end
        for elset in elsets
            println(io, "*ELSET,ELSET=" * elset)
            for i in 1:nfaces
                if i == nfaces
                    println(io, i)
                else
                    print(io, i, ", ")
                end
                if i % 10 == 0
                    println(io)
                end
            end
        end
    catch e
        println(e)
    finally
        close(io)
    end
    return nothing
end
