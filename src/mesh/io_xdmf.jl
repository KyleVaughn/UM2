function write_xdmf(mpt::MeshPartitionTree, filename::String)
    # Check valid filename
    if !endswith(filename, ".xdmf")
        error("Invalid filename.")
    end

    material_groups = String[]
    for mesh in leaves(mpt) 
        for group_name in keys(mesh.groups)
            if startswith(group_name, "Material") && group_name ∉ material_groups
                push!(material_groups, group_name)
            end
        end
    end
    # h5 filename
    h5_filename = filename[1:end-4]*"h5"
    h5_file = h5open(h5_filename, "w")
    # XML
    xdoc = XMLDocument()

    try
        # Xdmf
        xroot = create_root(xdoc, "Xdmf")
        set_attribute(xroot, "Version", "3.0")
        # Domain
        xdomain = new_child(xroot, "Domain")
        # Material names
        materials = [ mat[11:end] for mat in material_groups ]
        if materials !== String[] 
            xmaterials = new_child(xdomain, "Information")
            set_attribute(xmaterials, "Name", "Materials")
            add_text(xmaterials, join(materials, " ")) 
        end
        _add_mesh_partition_xdmf!(xdomain, h5_filename, h5_file, 
                                  mpt.partition_tree, mpt.leaf_meshes, material_groups)
    finally
        save_file(xdoc, filename)
        close(h5_file)
    end
    return nothing
end

function write_xdmf(mesh::PolytopeVertexMesh, filename::String) 
    # Check valid filename
    if !endswith(filename, ".xdmf")
        error("Invalid filename.")
    end

    # h5
    h5_filename = filename[1:end-4]*"h5"
    h5_file = h5open(h5_filename, "w")

    # XML
    xdoc = XMLDocument()
    try
        # Xdmf
        xroot = create_root(xdoc, "Xdmf")
        set_attribute(xroot, "Version", "3.0")
        # Domain
        xdomain = new_child(xroot, "Domain")
        
        material_groups = sort!(filter!(x->startswith(x, "Material"), 
                                        collect(keys(mesh.groups))
                                       )
                               )
        materials = [ mat[11:end] for mat in material_groups ]

        # Material names
        if materials !== String[] 
            xmaterials = new_child(xdomain, "Information")
            set_attribute(xmaterials, "Name", "Materials")
            add_text(xmaterials, join(materials, " ")) 
        end

    # Add uniform grid
    _add_uniform_grid_xdmf!(xdomain, h5_filename, h5_file, mesh, material_groups)

    finally
        save_file(xdoc, filename)
        close(h5_file)
    end
    return nothing
end

# Helper functions for write_xdmf
# -------------------------------------------------------------------------------------------------
function _add_uniform_grid_xdmf!(xml::XMLElement,
                                h5_filename::String,
                                h5_mesh::Union{HDF5.Group, HDF5.File},
                                mesh::PolytopeVertexMesh,
                                materials::Vector{String})
    # Grid                                                                       
    xgrid = new_child(xml, "Grid")
    set_attribute(xgrid, "Name", mesh.name)
    set_attribute(xgrid, "GridType", "Uniform")

    # h5
    h5_group = create_group(h5_mesh, mesh.name)
 
    # Geometry
    _write_xdmf_geometry!(xgrid, h5_filename, h5_group, mesh)
 
    # Topology
    _write_xdmf_topology!(xgrid, h5_filename, h5_group, mesh)
 
    # Non-material groups 
    if mesh.groups != Dict{String,BitSet}() 
        _write_xdmf_groups!(xgrid, h5_filename, h5_group, mesh)
    end
 
    # Materials
    if materials != String[] 
        _write_xdmf_materials!(xgrid, h5_filename, h5_group, mesh, materials)
    end
    return nothing
end

function _write_xdmf_geometry!(xml::XMLElement, 
                              h5_filename::String, 
                              h5_mesh::HDF5.Group, 
                              mesh::PolytopeVertexMesh)

    verts = vertices(mesh)
    nverts = length(verts)
    nverts_str = string(nverts)
    point_dim = length(verts[1])
    # Geometry
    xgeom = new_child(xml, "Geometry")
    if point_dim == 2 
        XYZ = "XY"
    else
        XYZ= "XYZ"
    end
    set_attribute(xgeom, "GeometryType", XYZ)
    # DataItem
    xdataitem = new_child(xgeom, "DataItem")
    set_attribute(xdataitem, "DataType", "Float")
    if point_dim == 2 
        dim = " 2"
    else
        dim = " 3"
    end
    set_attribute(xdataitem, "Dimensions", nverts_str*dim)
    set_attribute(xdataitem, "Format", "HDF")
    float_type = typeof(verts[1][1])
    if float_type === Float32
        float_precision = "4"
    elseif float_type === Float64
        float_precision = "8"
    else
        error("Could not determine float type.")
    end
    set_attribute(xdataitem, "Precision", float_precision)

    h5_text_item = split(string(h5_mesh))[2]
    add_text(xdataitem, string(h5_filename, ":", h5_text_item, "/vertices"))
    # Convert the points into an array
    vert_array = zeros(float_type, point_dim, nverts)
    for i in eachindex(verts)
        vert_array[:, i] = coordinates(verts[i])
    end
    # Write the h5
    h5_mesh["vertices"] = vert_array
    return nothing
end

function _write_xdmf_topology!(xml::XMLElement, 
                              h5_filename::String, 
                              h5_mesh::HDF5.Group, 
                              mesh::PolytopeVertexMesh)
    # Topology
    xtopo = new_child(xml, "Topology")
    set_attribute(xtopo, "TopologyType", "Mixed")
    ptopes = polytopes(mesh)
    nelements = length(ptopes)
    nelements_str = string(nelements)
    set_attribute(xtopo, "NumberOfElements", nelements_str)
    # DataItem
    xdataitem = new_child(xtopo, "DataItem")
    set_attribute(xdataitem, "DataType", "UInt")
    topo_length = mapreduce(x->length(vertices(x)), +, polytopes(mesh)) + nelements
    uint_type = typeof(ptopes[1][1])
    if uint_type === UInt8
        uint_precision = "1"
    elseif uint_type === UInt16
        uint_precision = "2"
    elseif uint_type === UInt32
        uint_precision = "4"
    elseif uint_type === UInt64
        uint_precision = "8"
    else
        error("Could not determine UInt type.")
    end
    topo_array = Vector{uint_type}(undef, topo_length)
    topo_ctr = 1
    for i in eachindex(ptopes)
        p = ptopes[i]
        if p isa Triangle
            topo_array[topo_ctr] = 4
        elseif p isa Quadrilateral
            topo_array[topo_ctr] = 5
        elseif p isa Tetrahedron
            topo_array[topo_ctr] = 6
        elseif p isa Hexahedron
            topo_array[topo_ctr] = 9
        elseif p isa QuadraticTriangle
            topo_array[topo_ctr] = 36
        elseif p isa QuadraticQuadrilateral
            topo_array[topo_ctr] = 37
        elseif p isa QuadraticTetrahedron
            topo_array[topo_ctr] = 38
        elseif p isa QuadraticHexahedron
            topo_array[topo_ctr] = 48
        else
            error("Could not determine topology type.")
        end
        topo_ctr += 1
        len_p = length(vertices(p))
        # adjust 1-based to 0-based indexing
        topo_array[topo_ctr:topo_ctr + len_p - 1] = vertices(p) .- 1 
        topo_ctr += len_p
    end
    ndims = string(length(topo_array))
    set_attribute(xdataitem, "Dimensions", ndims)
    set_attribute(xdataitem, "Format", "HDF")
    set_attribute(xdataitem, "Precision", uint_precision)
    h5_text_item = split(string(h5_mesh))[2]
    add_text(xdataitem, string(h5_filename, ":", h5_text_item, "/polytopes"))
    # Write the h5
    h5_mesh["polytopes"] = topo_array
    return nothing
end

function _write_xdmf_materials!(xml::XMLElement, 
                               h5_filename::String, 
                               h5_mesh::HDF5.Group, 
                               mesh::PolytopeVertexMesh,
                               materials::Vector{String})
    # MaterialID
    xmaterial = new_child(xml, "Attribute")
    set_attribute(xmaterial, "Center", "Cell")
    set_attribute(xmaterial, "Name", "Material")
    nelements = length(mesh.polytopes)
    mat_id_array = fill(-1, nelements)
    mesh_group_names = keys(mesh.groups)
    for (mat_id, material_name) in enumerate(materials)
        if material_name in mesh_group_names 
            for cell in mesh.groups[material_name]
                if mat_id_array[cell] === -1
                    mat_id_array[cell] = mat_id - 1 # 0 based indexing
                else
                    error("Mesh cell "*string(cell)*" has multiple materials assigned to it.")
                end
            end
        end
    end
    if any(x->x === -1, mat_id_array)
        error("Some mesh cells do not have a material.")
    end
    # DataItem
    xdataitem = new_child(xmaterial, "DataItem")
    set_attribute(xdataitem, "DataType", "UInt")
    set_attribute(xdataitem, "Dimensions", string(nelements))
    set_attribute(xdataitem, "Format", "HDF")
    N = length(materials)
    if N ≤ typemax(UInt8) 
        uint_type = UInt8
    elseif N ≤ typemax(UInt16) 
        uint_type = UInt16
    elseif N ≤ typemax(UInt32) 
        uint_type = UInt32
    elseif N ≤ typemax(UInt64) 
        uint_type = UInt64
    else
        error("N is not representable by UInt64.")
    end
    if uint_type === UInt8
        uint_precision = "1"
    elseif uint_type === UInt16
        uint_precision = "2"
    elseif uint_type === UInt32
        uint_precision = "4"
    elseif uint_type === UInt64
        uint_precision = "8"
    else
        error("Could not determine UInt type.")
    end

    set_attribute(xdataitem, "Precision", uint_precision)
    h5_text_item = split(string(h5_mesh))[2]
    add_text(xdataitem, string(h5_filename, ":", h5_text_item, "/material"))
    # Write the h5
    h5_mesh["material"] = mat_id_array
    return nothing
end

function _write_xdmf_groups!(xml::XMLElement, 
                             h5_filename::String, 
                             h5_mesh::HDF5.Group, 
                             mesh::PolytopeVertexMesh)
    for set_name in keys(mesh.groups)
        if startswith(set_name, "Material")
            continue
        end
        grp = mesh.groups[set_name]
        # Set
        xset = new_child(xml, "Set")
        set_attribute(xset, "Name", set_name)
        set_attribute(xset, "SetType", "Cell")
        # DataItem
        xdataitem = new_child(xset, "DataItem")
        set_attribute(xdataitem, "DataType", "UInt")
        nelements = string(length(grp))
        set_attribute(xdataitem, "Dimensions", nelements)
        set_attribute(xdataitem, "Format", "HDF")
        N = maximum(grp)
        if N ≤ typemax(UInt8) 
            uint_type = UInt8
        elseif N ≤ typemax(UInt16) 
            uint_type = UInt16
        elseif N ≤ typemax(UInt32) 
            uint_type = UInt32
        elseif N ≤ typemax(UInt64) 
            uint_type = UInt64
        else
            error("N is not representable by UInt64.")
        end
        if uint_type === UInt8
            uint_precision = "1"
        elseif uint_type === UInt16
            uint_precision = "2"
        elseif uint_type === UInt32
            uint_precision = "4"
        elseif uint_type === UInt64
            uint_precision = "8"
        else
            error("Could not determine UInt type.")
        end
        set_attribute(xdataitem, "Precision", uint_precision)
        h5_text_item = split(string(h5_mesh))[2]
        add_text(xdataitem, string(h5_filename, ":", h5_text_item, "/", set_name))
        # Write the h5
        ID_array = collect(grp) .- 1 
        h5_mesh[set_name] = ID_array
    end
    return nothing
end

function _add_mesh_partition_xdmf!(xml::XMLElement,
                                   h5_filename::String,
                                   h5_mesh::Union{HDF5.Group, HDF5.File},
                                   node::Tree{Tuple{Int64, String}},
                                   leaf_meshes::Vector{<:PolytopeVertexMesh},
                                   materials::Vector{String})
    id = node.data[1]
    if id === 0 # Internal node
        name = node.data[2]
        # Grid
        xgrid = new_child(xml, "Grid")
        set_attribute(xgrid, "Name", name)
        set_attribute(xgrid, "GridType", "Tree")
        # h5_group
        h5_group = create_group(h5_mesh, name)
        for child in node.children
            _add_mesh_partition_xdmf!(xgrid, h5_filename, h5_group, child, 
                                      leaf_meshes, materials)
        end
    else # Leaf node 
        _add_uniform_grid_xdmf!(xml, h5_filename, h5_mesh, 
                                leaf_meshes[id], materials)
    end
    return nothing
end
