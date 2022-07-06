function setup_Triangle_PVM(::Type{T}, ::Type{U}) where {T, U}
    name = "tri"
    vertices = [Point{2, T}(0, 0),
        Point{2, T}(1, 0),
        Point{2, T}(1 // 2, 1),
        Point{2, T}(3 // 2, 1)]
    polytopes = [Triangle{U}(1, 2, 3), Triangle{U}(2, 4, 3)]
    materials = UInt8[2, 1]
    material_names = ["H2O", "UO2"]
    groups = Dict{String, BitSet}()
    groups["A"] = BitSet([1])
    groups["B"] = BitSet([2])
    return PolytopeVertexMesh(vertices, polytopes, materials, material_names,
                              name, groups)
end

function setup_Quadrilateral_PVM(::Type{T}, ::Type{U}) where {T, U}
    name = "quad"
    vertices = [Point{2, T}(0, 0),
        Point{2, T}(1, 0),
        Point{2, T}(1, 1),
        Point{2, T}(0, 1),
        Point{2, T}(2, 0),
        Point{2, T}(2, 1)]
    polytopes = [Quadrilateral{U}(1, 2, 3, 4), Quadrilateral{U}(2, 5, 6, 3)]
    materials = UInt8[2, 1]
    material_names = ["H2O", "UO2"]
    groups = Dict{String, BitSet}()
    groups["A"] = BitSet([1])
    groups["B"] = BitSet([2])
    return PolytopeVertexMesh(vertices, polytopes, materials, material_names,
                              name, groups)
end

function setup_MixedPolygon_PVM(::Type{T}, ::Type{U}) where {T, U}
    name = "tri_quad"
    vertices = [Point{2, T}(0, 0),
        Point{2, T}(1, 0),
        Point{2, T}(1, 1),
        Point{2, T}(0, 1),
        Point{2, T}(2, 0)]
    polytopes = [Quadrilateral{U}(1, 2, 3, 4), Triangle{U}(2, 5, 3)]
    materials = UInt8[2, 1]
    material_names = ["H2O", "UO2"]
    groups = Dict{String, BitSet}()
    groups["A"] = BitSet([1])
    groups["B"] = BitSet([2])
    return PolytopeVertexMesh(vertices, polytopes, materials, material_names,
                              name, groups)
end

function setup_QuadraticTriangle_PVM(::Type{T}, ::Type{U}) where {T, U}
    name = "tri6"
    vertices = [Point{2, T}(0, 0),
        Point{2, T}(1, 0),
        Point{2, T}(0, 1),
        Point{2, T}(1 // 2, 0),
        Point{2, T}(7 // 16, 7 // 16),
        Point{2, T}(0, 1 // 2),
        Point{2, T}(1, 1),
        Point{2, T}(1, 1 // 2),
        Point{2, T}(1 // 2, 1)]
    polytopes = [QuadraticTriangle{U}(1, 2, 3, 4, 5, 6),
        QuadraticTriangle{U}(2, 7, 3, 8, 9, 5)]
    materials = UInt8[2, 1]
    material_names = ["H2O", "UO2"]
    groups = Dict{String, BitSet}()
    groups["A"] = BitSet([1])
    groups["B"] = BitSet([2])
    return PolytopeVertexMesh(vertices, polytopes, materials, material_names,
                              name, groups)
end

function setup_QuadraticQuadrilateral_PVM(::Type{T}, ::Type{U}) where {T, U}
    name = "quad8"
    vertices = [Point{2, T}(0, 0),
        Point{2, T}(1, 0),
        Point{2, T}(1, 1),
        Point{2, T}(0, 1),
        Point{2, T}(2, 0),
        Point{2, T}(2, 1),
        Point{2, T}(1 // 2, 0),
        Point{2, T}(9 // 16, 1 // 2),
        Point{2, T}(1 // 2, 1),
        Point{2, T}(0, 1 // 2),
        Point{2, T}(3 // 2, 0),
        Point{2, T}(2, 1 // 2),
        Point{2, T}(3 // 2, 1),
    ]
    polytopes = [QuadraticQuadrilateral{U}(1, 2, 3, 4, 7, 8, 9, 10),
        QuadraticQuadrilateral{U}(2, 5, 6, 3, 11, 12, 13, 8)]
    materials = UInt8[2, 1]
    material_names = ["H2O", "UO2"]
    groups = Dict{String, BitSet}()
    groups["A"] = BitSet([1])
    groups["B"] = BitSet([2])
    return PolytopeVertexMesh(vertices, polytopes, materials, material_names,
                              name, groups)
end

function setup_MixedQuadraticPolygon_PVM(::Type{T}, ::Type{U}) where {T, U}
    name = "tri6_quad8"
    vertices = [Point{2, T}(0, 0),
        Point{2, T}(1, 0),
        Point{2, T}(1, 1),
        Point{2, T}(0, 1),
        Point{2, T}(2, 0),
        Point{2, T}(1 // 2, 0),
        Point{2, T}(9 // 16, 1 // 2),
        Point{2, T}(1 // 2, 1),
        Point{2, T}(0, 1 // 2),
        Point{2, T}(3 // 2, 0),
        Point{2, T}(3 // 2, 1 // 2),
    ]
    polytopes = [QuadraticQuadrilateral{U}(1, 2, 3, 4, 6, 7, 8, 9),
        QuadraticTriangle{U}(2, 5, 3, 10, 11, 7)]
    materials = UInt8[2, 1]
    material_names = ["H2O", "UO2"]
    groups = Dict{String, BitSet}()
    groups["A"] = BitSet([1])
    groups["B"] = BitSet([2])
    return PolytopeVertexMesh(vertices, polytopes, materials, material_names,
                              name, groups)
end

###########################################################################3
function setup_Triangle_VM(::Type{T}, ::Type{U}) where {T, U}
    name = "tri"
    points = [Point{2, T}(0, 0),
        Point{2, T}(1, 0),
        Point{2, T}(1 // 2, 1),
        Point{2, T}(3 // 2, 1)]
    offsets = U[1, 4, 7]
    connectivity = U[1, 2, 3, 2, 4, 3]
    materials = UInt8[2, 1]
    material_names = ["H2O", "UO2"]
    groups = Dict{String, BitSet}()
    groups["A"] = BitSet([1])
    groups["B"] = BitSet([2])
    return VolumeMesh(points, offsets, connectivity, materials, 
                      material_names, name, groups)
end

function setup_Quadrilateral_VM(::Type{T}, ::Type{U}) where {T, U}
    name = "quad"
    points = [Point{2, T}(0, 0),
        Point{2, T}(1, 0),
        Point{2, T}(1, 1),
        Point{2, T}(0, 1),
        Point{2, T}(2, 0),
        Point{2, T}(2, 1)]
    offsets = U[1, 5, 9]
    connectivity = U[1, 2, 3, 4, 2, 5, 6, 3]
    materials = UInt8[2, 1]
    material_names = ["H2O", "UO2"]
    groups = Dict{String, BitSet}()
    groups["A"] = BitSet([1])
    groups["B"] = BitSet([2])
    return VolumeMesh(points, offsets, connectivity, materials, 
                      material_names, name, groups)
end

function setup_MixedPolygon_VM(::Type{T}, ::Type{U}) where {T, U}
    name = "tri_quad"
    points = [Point{2, T}(0, 0),
        Point{2, T}(1, 0),
        Point{2, T}(1, 1),
        Point{2, T}(0, 1),
        Point{2, T}(2, 0)]
    offsets = U[1, 5, 8]
    connectivity = U[1, 2, 3, 4, 2, 5, 3]
    materials = UInt8[2, 1]
    material_names = ["H2O", "UO2"]
    groups = Dict{String, BitSet}()
    groups["A"] = BitSet([1])
    groups["B"] = BitSet([2])
    return VolumeMesh(points, offsets, connectivity, materials,
                      material_names, name, groups)
end

function setup_QuadraticTriangle_VM(::Type{T}, ::Type{U}) where {T, U}
    name = "tri6"
    points = [Point{2, T}(0, 0),
        Point{2, T}(1, 0),
        Point{2, T}(0, 1),
        Point{2, T}(1 // 2, 0),
        Point{2, T}(7 // 16, 7 // 16),
        Point{2, T}(0, 1 // 2),
        Point{2, T}(1, 1),
        Point{2, T}(1, 1 // 2),
        Point{2, T}(1 // 2, 1)]
    offsets = U[1, 7, 13]
    connectivity = U[1, 2, 3, 4, 5, 6, 2, 7, 3, 8, 9, 5]
    materials = UInt8[2, 1]
    material_names = ["H2O", "UO2"]
    groups = Dict{String, BitSet}()
    groups["A"] = BitSet([1])
    groups["B"] = BitSet([2])
    return VolumeMesh(points, offsets, connectivity, materials, 
                      material_names, name, groups)
end

function setup_QuadraticQuadrilateral_VM(::Type{T}, ::Type{U}) where {T, U}
    name = "quad8"
    points = [Point{2, T}(0, 0),
        Point{2, T}(1, 0),
        Point{2, T}(1, 1),
        Point{2, T}(0, 1),
        Point{2, T}(2, 0),
        Point{2, T}(2, 1),
        Point{2, T}(1 // 2, 0),
        Point{2, T}(9 // 16, 9 // 16),
        Point{2, T}(1 // 2, 1),
        Point{2, T}(0, 1 // 2),
        Point{2, T}(3 // 2, 0),
        Point{2, T}(2, 1 // 2),
        Point{2, T}(3 // 2, 1),
    ]
    offsets = U[1, 9, 17]
    connectivity = U[1, 2, 3, 4, 7, 8, 9, 10, 2, 5, 6, 3, 11, 12, 13, 8]
    materials = UInt8[2, 1]
    material_names = ["H2O", "UO2"]
    groups = Dict{String, BitSet}()
    groups["A"] = BitSet([1])
    groups["B"] = BitSet([2])
    return VolumeMesh(points, offsets, connectivity, materials, 
                      material_names, name, groups)
end

function setup_MixedQuadraticPolygon_VM(::Type{T}, ::Type{U}) where {T, U}
    name = "tri6_quad8"
    points = [Point{2, T}(0, 0),
        Point{2, T}(1, 0),
        Point{2, T}(1, 1),
        Point{2, T}(0, 1),
        Point{2, T}(2, 0),
        Point{2, T}(1 // 2, 0),
        Point{2, T}(9 // 16, 1 // 2),
        Point{2, T}(1 // 2, 1),
        Point{2, T}(0, 1 // 2),
        Point{2, T}(3 // 2, 0),
        Point{2, T}(3 // 2, 1 // 2),
    ]
    offsets = U[1, 9, 15]
    connectivity = U[1, 2, 3, 4, 6, 7, 8, 9, 2, 5, 3, 10, 11, 7]
    materials = UInt8[2, 1]
    material_names = ["H2O", "UO2"]
    groups = Dict{String, BitSet}()
    groups["A"] = BitSet([1])
    groups["B"] = BitSet([2])
    return VolumeMesh(points, offsets, connectivity, materials, material_names,
                              name, groups)
end
