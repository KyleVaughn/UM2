const vtk_to_xdmf_type = Dict(
    # triangle
    5  => 4,
    # triangle6
    22 => 36, 
    # quadrilateral
    9 => 5,
    # quad8
    23 => 37
   )  

function write_xdmf_2d(filename::String, mesh::UnstructuredMesh_2D)
    @info "Writing $filename" 
    # Check valid filename
    if !occursin(".xdmf", filename)
        error("Invalid filename. '.xdmf' does not occur in $filename") 
    end

    # If there are materials, map all material names to an integer
    material_map = Dict{String, Int64}()
    if mesh.face_sets != Dict{String, Set{Int64}}()
        material_map = _make_material_name_to_id_map(mesh)
    end

    # h5 filename
    h5_filename = replace(filename, ("xdmf" => "h5"))
    h5_file = h5open(h5_filename, "w")
    h5_mesh = create_group(h5_file, mesh.name)
    # XML
    xdoc = XMLDocument()
    # Xdmf
    xroot = create_root(xdoc, "Xdmf")
    set_attribute(xroot, "Version", "3.0")
    # Domain
    xdomain = new_child(xroot, "Domain")
    # Material names
    if material_map != Dict{String, Int64}()
        xmaterials = new_child(xdomain, "Information")
        set_attribute(xmaterials, "Name", "MaterialNames")
        add_text(xmaterials, join(keys(material_map), " ")) 
    end
    # Grid
    xgrid = new_child(xdomain, "Grid")
    set_attribute(xgrid, "Name", mesh.name)
    set_attribute(xgrid, "GridType", "Uniform")

    # Geometry
    _write_xdmf_geometry(xgrid, h5_filename, h5_mesh, mesh)

    # Topology
    _write_xdmf_topology(xgrid, h5_filename, h5_mesh, mesh)

    # Non-material face sets
    if mesh.face_sets != Dict{String, Set{Int64}}()
        _write_xdmf_face_sets(xgrid, h5_filename, h5_mesh, mesh)
    end

    # Materials
    if material_map != Dict{String, Int64}()
        _write_xdmf_materials(xgrid, h5_filename, h5_mesh, mesh, material_map)
    end

    save_file(xdoc, filename)
    close(h5_file)
end

function _write_xdmf_geometry(xml::XMLElement, 
                              h5_filename::String, 
                              h5_mesh::HDF5.Group, 
                              mesh::UnstructuredMesh_2D)
    # Geometry
    xgeom = new_child(xml, "Geometry")
    set_attribute(xgeom, "GeometryType", "XYZ")
    # DataItem
    xdataitem = new_child(xgeom, "DataItem")
    set_attribute(xdataitem, "DataType", "Float")
    npoints = length(mesh.points)
    set_attribute(xdataitem, "Dimensions", "$npoints 3")
    set_attribute(xdataitem, "Format", "HDF")
    float_precision = 8
    float_type = Float64
    if typeof(mesh.points[1]) === Point_2D{Float32}
        float_precision = 4
        float_type = Float32
    end
    set_attribute(xdataitem, "Precision", "$float_precision")
    add_text(xdataitem, string(h5_filename, ":/", mesh.name, "/points"))
    # Convert the points into an array
    point_array = zeros(float_type, npoints, 3)
    for i = 1:npoints
        point_array[i, 1] = mesh.points[i][1]
        point_array[i, 2] = mesh.points[i][2]
    end
    point_array = copy(transpose(point_array))
    # Write the h5
    h5_mesh["points"] = point_array
end

function _write_xdmf_topology(xml::XMLElement, 
                              h5_filename::String, 
                              h5_mesh::HDF5.Group, 
                              mesh::UnstructuredMesh_2D)
    # Topology
    xtopo = new_child(xml, "Topology")
    set_attribute(xtopo, "TopologyType", "Mixed")
    nelements = length(mesh.faces)
    set_attribute(xtopo, "NumberOfElements", "$nelements")
    # DataItem
    xdataitem = new_child(xtopo, "DataItem")
    set_attribute(xdataitem, "DataType", "Int")
    topo_array = Int64[]
    for face in mesh.faces
        # convert face to vector for mutability
        face_xdmf = [x for x in face]
        # adjust vtk to xdmf type
        face_xdmf[1] = vtk_to_xdmf_type[face_xdmf[1]]
        # adjust 1-based to 0-based indexing
        face_xdmf[2:length(face_xdmf)] = face_xdmf[2:length(face_xdmf)] .- 1
        for value in face_xdmf
            push!(topo_array, value) 
        end
    end
    ndimensions = length(topo_array)
    set_attribute(xdataitem, "Dimensions", "$ndimensions")
    set_attribute(xdataitem, "Format", "HDF")
    set_attribute(xdataitem, "Precision", "8")
    add_text(xdataitem, string(h5_filename, ":/", mesh.name, "/cells"))
    # Write the h5
    h5_mesh["cells"] = topo_array
end

function _make_material_name_to_id_map(mesh::UnstructuredMesh_2D)
    material_map = Dict{String, Int64}()
    nmat = 0
    for set_name in keys(mesh.face_sets)
        if occursin("MATERIAL", uppercase(set_name))
            material_map[set_name] = nmat 
            nmat += 1
        end
    end
    return material_map
end

function _write_xdmf_materials(xml::XMLElement, 
                               h5_filename::String, 
                               h5_mesh::HDF5.Group, 
                               mesh::UnstructuredMesh_2D,
                               material_map::Dict{String, Int64})
    # MaterialID
    xmaterial = new_child(xml, "Attribute")
    set_attribute(xmaterial, "Center", "Cell")
    set_attribute(xmaterial, "Name", "MaterialID")
    nelements = length(mesh.faces)
    mat_ID_array = zeros(Int64, nelements) .- 1
    for material_name in keys(material_map)
        material_ID = material_map[material_name]
        for cell in mesh.face_sets[material_name]
            if mat_ID_array[cell] === -1
                mat_ID_array[cell] = material_ID
            else
                error("Mesh cell $cell has multiple materials assigned to it.")
            end
        end
    end
    if any(x->x === -1, mat_ID_array)
        error("Some mesh cells do not have a material.")
    end
    # DataItem
    xdataitem = new_child(xmaterial, "DataItem")
    set_attribute(xdataitem, "DataType", "Int")
    set_attribute(xdataitem, "Dimensions", "$nelements")
    set_attribute(xdataitem, "Format", "HDF")
    set_attribute(xdataitem, "Precision", "8")
    add_text(xdataitem, string(h5_filename, ":/", mesh.name, "/material_id"))
    # Write the h5
    h5_mesh["material_id"] = mat_ID_array
end

function _write_xdmf_face_sets(xml::XMLElement, 
                               h5_filename::String, 
                               h5_mesh::HDF5.Group, 
                               mesh::UnstructuredMesh_2D)
    for set_name in keys(mesh.face_sets)
        if occursin("MATERIAL", uppercase(set_name))
            continue
        end
        # Set
        xset = new_child(xml, "Set")
        set_attribute(xset, "Name", "$set_name")
        set_attribute(xset, "SetType", "Cell")
        # DataItem
        xdataitem = new_child(xset, "DataItem")
        set_attribute(xdataitem, "DataType", "Int")
        nelements = length(mesh.face_sets[set_name])
        set_attribute(xdataitem, "Dimensions", "$nelements")
        set_attribute(xdataitem, "Format", "HDF")
        set_attribute(xdataitem, "Precision", "8")
        add_text(xdataitem, string(h5_filename, ":/", mesh.name, "/$set_name"))
        # Write the h5
        ID_array = [ x - 1 for x in mesh.face_sets[set_name] ] 
        h5_mesh["$set_name"] = ID_array
    end
end
