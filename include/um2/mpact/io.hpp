#pragma once

// TODO: Split repeated code into functions. There's a lot of repeated code

#include <um2/mesh/io_xdmf.hpp>
#include <um2/mpact/spatial_partition.hpp>

namespace um2
{

template <std::floating_point T, std::signed_integral I>
void
write_xdmf_file(std::string const & path, mpact::SpatialPartition<T, I> & model)
{
  Log::info("Writing XDMF file for MPACT model: " + path);

  size_t const h5filepath_end = path.find_last_of("/") + 1;
  std::string name = path.substr(h5filepath_end, path.size() - 5 - h5filepath_end);
  std::string h5filename = name + ".h5";
  std::string h5filepath = path.substr(0, h5filepath_end);
  um2::Log::debug("H5 filename: " + h5filename);
  H5::H5File h5file(h5filepath + h5filename, H5F_ACC_TRUNC);
  std::string h5path = "";

  // Setup XML file
  pugi::xml_document xdoc;

  // XDMF root node
  pugi::xml_node xroot = xdoc.append_child("Xdmf");
  xroot.append_attribute("Version") = "3.0";

  // Domain node
  pugi::xml_node xdomain = xroot.append_child("Domain");

  // Material info
  pugi::xml_node xinfo = xdomain.append_child("Information");
  xinfo.append_attribute("Name") = "Materials";
  std::string materials = "";
  Vector<std::string> mat_names;
  Vector<std::string> mat_names_short;
  std::string const material_str = "Material_";
  for (length_t i = 0; i < model.materials.size(); i++) {
    materials += to_string(model.materials[i].name);
    if (i + 1 < model.materials.size()) {
      materials += ", ";
    }
    mat_names.push_back(material_str + to_string(model.materials[i].name));
    mat_names_short.push_back(to_string(model.materials[i].name));
  }
  xinfo.append_child(pugi::node_pcdata).set_value(materials.c_str());

  // Core grid
  pugi::xml_node xcore_grid = xdomain.append_child("Grid");
  xcore_grid.append_attribute("Name") = name.c_str();
  xcore_grid.append_attribute("GridType") = "Tree";

  // h5
  H5::Group h5core_group = h5file.createGroup(name);
  std::string h5core_grouppath = h5path + "/" + name;

  Vector<I> cc_found(model.coarse_cells.size(), -1);
  Vector<I> rtm_found(model.rtms.size(), -1);
  Vector<I> lat_found(model.lattices.size(), -1);
  Vector<I> ass_found(model.assemblies.size(), -1);
  auto const & core = model.core;
  if (core.children.empty()) {
    Log::error("Core has no children");
    return;
  }
  std::stringstream ss;
  length_t const nyass = num_ycells(core);
  length_t const nxass = num_xcells(core);
  // Core M by N
  pugi::xml_node xcore_info = xcore_grid.append_child("Information");
  xcore_info.append_attribute("Name") = "M_by_N";
  std::string core_mn_str = std::to_string(nyass) + " x " + std::to_string(nxass);
  xcore_info.append_child(pugi::node_pcdata).set_value(core_mn_str.c_str());
  // For each assembly
  for (length_t iyass = 0; iyass < nyass; ++iyass) {
    for (length_t ixass = 0; ixass < nxass; ++ixass) {
      I const ass_id = core.get_child(ixass, iyass);
      ass_found[static_cast<length_t>(ass_id)] += 1;
      I const ass_id_ctr = ass_found[static_cast<length_t>(ass_id)];
      pugi::xml_node xass_grid = xcore_grid.append_child("Grid");
      ss.str("");
      ss << "Assembly_" << std::setw(5) << std::setfill('0') << ass_id << "_"
         << std::setw(5) << std::setfill('0') << ass_id_ctr;
      xass_grid.append_attribute("Name") = ss.str().c_str();
      xass_grid.append_attribute("GridType") = "Tree";
      std::string h5ass_grouppath = h5core_grouppath + "/" + ss.str();
      H5::Group h5ass_group = h5file.createGroup(h5ass_grouppath);
      AABox2<T> const ass_bb = core.get_box(ixass, iyass);
      Point2<T> ass_ll = ass_bb.minima; // Lower left corner
      auto const & assembly = model.assemblies[ass_id];
      if (assembly.children.empty()) {
        Log::error("Assembly has no children");
        return;
      }
      length_t const nzlat = num_cells(assembly)[0];
      // Assembly M by N
      pugi::xml_node xass_info = xass_grid.append_child("Information");
      xass_info.append_attribute("Name") = "M_by_N";
      std::string ass_mn_str = std::to_string(nzlat) + " x 1";
      xass_info.append_child(pugi::node_pcdata).set_value(ass_mn_str.c_str());
      // For each lattice
      for (length_t izlat = 0; izlat < nzlat; ++izlat) {
        I const lat_id = assembly.children[izlat];
        lat_found[static_cast<length_t>(lat_id)] += 1;
        I const lat_id_ctr = lat_found[static_cast<length_t>(lat_id)];
        pugi::xml_node xlat_grid = xass_grid.append_child("Grid");
        ss.str("");
        ss << "Lattice_" << std::setw(5) << std::setfill('0') << lat_id << "_"
           << std::setw(5) << std::setfill('0') << lat_id_ctr;
        xlat_grid.append_attribute("Name") = ss.str().c_str();
        xlat_grid.append_attribute("GridType") = "Tree";
        std::string h5lat_grouppath = h5ass_grouppath + "/" + ss.str();
        H5::Group h5lat_group = h5file.createGroup(h5lat_grouppath);
        T const low_z = assembly.grid.divs[0][izlat];
        T const high_z = assembly.grid.divs[0][izlat + 1];
        T const cut_z = (low_z + high_z) / 2;
        pugi::xml_node xlat_info = xlat_grid.append_child("Information");
        xlat_info.append_attribute("Name") = "Z";
        std::string z_values = std::to_string(low_z) + ", " + std::to_string(cut_z) +
                               ", " + std::to_string(high_z);
        xlat_info.append_child(pugi::node_pcdata).set_value(z_values.c_str());
        //            T const dz = (high_z - low_z) / 2;
        auto const & lattice = model.lattices[lat_id];
        if (lattice.children.empty()) {
          Log::error("Lattice has no children");
          return;
        }
        length_t const nyrtm = num_ycells(lattice);
        length_t const nxrtm = num_xcells(lattice);
        // Lattice M by N
        pugi::xml_node xlat_info2 = xlat_grid.append_child("Information");
        xlat_info2.append_attribute("Name") = "M_by_N";
        std::string lat_mn_str = std::to_string(nyrtm) + " x " + std::to_string(nxrtm);
        xlat_info2.append_child(pugi::node_pcdata).set_value(lat_mn_str.c_str());
        // For each RTM
        for (length_t iyrtm = 0; iyrtm < nyrtm; ++iyrtm) {
          for (length_t ixrtm = 0; ixrtm < nxrtm; ++ixrtm) {
            I const rtm_id = lattice.get_child(ixrtm, iyrtm);
            rtm_found[static_cast<length_t>(rtm_id)] += 1;
            I const rtm_id_ctr = rtm_found[static_cast<length_t>(rtm_id)];
            pugi::xml_node xrtm_grid = xlat_grid.append_child("Grid");
            ss.str("");
            ss << "RTM_" << std::setw(5) << std::setfill('0') << rtm_id << "_"
               << std::setw(5) << std::setfill('0') << rtm_id_ctr;
            xrtm_grid.append_attribute("Name") = ss.str().c_str();
            xrtm_grid.append_attribute("GridType") = "Tree";
            std::string h5rtm_grouppath = h5lat_grouppath + "/" + ss.str();
            H5::Group h5rtm_group = h5file.createGroup(h5rtm_grouppath);
            AABox2<T> const rtm_bb = lattice.get_box(ixrtm, iyrtm);
            Point2<T> rtm_ll = rtm_bb.minima; // Lower left corner
            auto const & rtm = model.rtms[rtm_id];
            if (rtm.children.empty()) {
              Log::error("RTM has no children");
              return;
            }
            length_t const nycells = num_ycells(rtm);
            length_t const nxcells = num_xcells(rtm);
            // RTM M by N
            pugi::xml_node xrtm_info = xrtm_grid.append_child("Information");
            xrtm_info.append_attribute("Name") = "M_by_N";
            std::string rtm_mn_str =
                std::to_string(nycells) + " x " + std::to_string(nxcells);
            xrtm_info.append_child(pugi::node_pcdata).set_value(rtm_mn_str.c_str());
            // For each coarse cell
            for (length_t iycell = 0; iycell < nycells; ++iycell) {
              for (length_t ixcell = 0; ixcell < nxcells; ++ixcell) {
                I const cell_id = rtm.get_child(ixcell, iycell);
                cc_found[static_cast<length_t>(cell_id)] += 1;
                AABox2<T> const cell_bb = rtm.get_box(ixcell, iycell);
                Point2<T> ll = ass_ll + rtm_ll + cell_bb.minima;
                I const cell_id_ctr = cc_found[static_cast<length_t>(cell_id)];
                I const mesh_type = model.coarse_cells[cell_id].mesh_type;
                I const mesh_id = model.coarse_cells[cell_id].mesh_id;
                MeshFile<T, I> mesh_file;
                switch (mesh_type) {
                case 1: // Triangle
                  model.tri[static_cast<length_t>(mesh_id)].to_mesh_file(mesh_file);
                  break;
                case 2: // Quadrilateral
                  model.quad[static_cast<length_t>(mesh_id)].to_mesh_file(mesh_file);
                  break;
                case 3: // Tri/Quad
                  model.tri_quad[static_cast<length_t>(mesh_id)].to_mesh_file(mesh_file);
                  break;
                case 4: // Quadratic Triangle
                  model.quadratic_tri[static_cast<length_t>(mesh_id)].to_mesh_file(
                      mesh_file);
                  break;
                case 5: // Quadratic Quadrilateral
                  model.quadratic_quad[static_cast<length_t>(mesh_id)].to_mesh_file(
                      mesh_file);
                  break;
                case 6: // Quadratic Tri/Quad
                  model.quadratic_tri_quad[static_cast<length_t>(mesh_id)].to_mesh_file(
                      mesh_file);
                  break;
                default:
                  Log::error("Unsupported mesh type");
                  return;
                } // switch
                // We need to add the material_ids as elsets to the mesh file
                // Get the number of materials in the mesh
                Vector<MaterialID> const & mat_ids =
                    model.coarse_cells[cell_id].material_ids;
                Vector<MaterialID> unique_mat_ids;
                for (auto const & mat_id : mat_ids) {
                  if (!unique_mat_ids.contains(mat_id)) {
                    unique_mat_ids.push_back(mat_id);
                  }
                }
                length_t const num_mats = unique_mat_ids.size();
                std::sort(unique_mat_ids.begin(), unique_mat_ids.end());
                mesh_file.elset_names.resize(num_mats);
                mesh_file.elset_offsets.resize(num_mats + 1);
                mesh_file.elset_offsets[0] = 0;
                length_t const num_elements = mesh_file.element_types.size();
                length_t elem_ctr = 0;
                mesh_file.elset_ids.resize(num_elements);
                for (length_t imat = 0; imat < num_mats; ++imat) {
                  MaterialID const mat_id = unique_mat_ids[imat];
                  mesh_file.elset_names[imat] = mat_names[mat_id];
                  for (length_t ielem = 0; ielem < num_elements; ++ielem) {
                    if (mat_ids[ielem] == mat_id) {
                      mesh_file.elset_ids[elem_ctr++] = ielem;
                      mesh_file.elset_offsets[imat + 1] = elem_ctr;
                    }
                  }
                }
                // Shift the mesh to global coordinates
                length_t const num_nodes = mesh_file.nodes_x.size();
                for (length_t inode = 0; inode < num_nodes; ++inode) {
                  mesh_file.nodes_x[inode] += ll[0];
                  mesh_file.nodes_y[inode] += ll[1];
                  mesh_file.nodes_z[inode] += cut_z;
                }
                ss.str("");
                ss << "Coarse_Cell_" << std::setw(5) << std::setfill('0') << cell_id
                   << "_" << std::setw(5) << std::setfill('0') << cell_id_ctr;
                mesh_file.name = ss.str();
                // Now write the coarse cell as a uniform grid
                write_xdmf_uniform_grid(xrtm_grid, h5file, h5filename, h5rtm_grouppath,
                                        mesh_file, mat_names_short);
              } // cell
            }   // cell

          } // rtm
        }   // rtm
      }     // lat
    }       // assembly
  }         // assembly
  // Write the XML file
  xdoc.save_file(path.c_str(), "  ");

  // Close the HDF5 file
  h5file.close();
}

template <std::floating_point T, std::signed_integral I>
void
export_mesh(std::string const & path, mpact::SpatialPartition<T, I> & model)
{
  if (path.ends_with(".xdmf")) {
    write_xdmf_file<T, I>(path, model);
  } else {
    Log::error("Unsupported file format.");
  }
}

static inline void
map_lattice_idx_to_j_i(int const idx, int & j, int & i, int const M, int const N)
{
  j = M - idx / N - 1;
  i = idx % N;
}

template <std::floating_point T, std::signed_integral I>
void
import_mpact_model(std::string const & path, mpact::SpatialPartition<T, I> & model)
{
  Log::info("Importing MPACT model from file: " + path);
  if (!path.ends_with(".xdmf")) {
    Log::error("Unsupported file format.");
    return;
  }

  // Open the XDMF file
  std::ifstream file(path);
  if (!file.is_open()) {
    Log::error("Could not open file: " + path);
    return;
  }
  // Open the HDF5 file
  size_t const h5filepath_end = path.find_last_of("/") + 1;
  std::string h5filename =
      path.substr(h5filepath_end, path.size() - 4 - h5filepath_end) + "h5";
  std::string h5filepath = path.substr(0, h5filepath_end);
  um2::Log::debug("H5 filename: " + h5filename);
  H5::H5File h5file(h5filepath + h5filename, H5F_ACC_RDONLY);

  // Setup XML document
  pugi::xml_document xdoc;
  pugi::xml_parse_result result = xdoc.load_file(path.c_str());
  if (!result) {
    Log::error("XDMF XML parse error: " + std::string(result.description()) +
               ", character pos= " + std::to_string(result.offset));
  }
  pugi::xml_node xroot = xdoc.child("Xdmf");
  if (strcmp("Xdmf", xroot.name()) != 0) {
    Log::error("XDMF XML root node is not Xdmf");
    return;
  }
  pugi::xml_node xdomain = xroot.child("Domain");
  if (strcmp("Domain", xdomain.name()) != 0) {
    Log::error("XDMF XML domain node is not Domain");
    return;
  }
  Vector<String> material_names;
  pugi::xml_node xinfo = xdomain.child("Information");
  if (strcmp("Information", xinfo.name()) == 0) {
    // Get the "Name" attribute
    pugi::xml_attribute xname = xinfo.attribute("Name");
    if (strcmp("Materials", xname.value()) == 0) {
      // Get the material names
      std::string materials = xinfo.child_value();
      std::stringstream ss(materials);
      std::string material;
      while (std::getline(ss, material, ',')) {
        if (material[0] == ' ') {
          material_names.push_back(material.substr(1));
        } else {
          material_names.push_back(material);
        }
      }
    }
  }
  model.materials.resize(material_names.size());
  for (length_t imat = 0; imat < material_names.size(); ++imat) {
    model.materials[imat].name = material_names[imat];
  }
  Vector<String> long_material_names(material_names.size());
  for (length_t imat = 0; imat < material_names.size(); ++imat) {
    long_material_names[imat] = "Material_" + to_string(material_names[imat]);
  }

  // Algorithm for populating the model:
  // -------------------------------
  // Constraints:
  // - We want to use the SpatialPartition.make_core, make_assembly, make_lattice,
  // make_rtm,
  //      and make_coarse_cell methods to create the model in order to avoid duplicating
  //      code. These functions take children IDs as arguments.
  // - We have to create ID 1 before ID 2, ID 2 before ID 3, etc. since the make_* methods
  //      do not take its own ID as an argument. Therefore we have to construct the model
  //      in a bottom-up fashion, i.e. we have to create the coarse cells before we can
  //      create the RTMs, etc.
  // - We want to avoid making multiple passes over the XDMF file.
  //
  // Algorithm:
  // Get the core node
  // Get the M by N size of the core
  // Allocate core_assembly_ids to be M by N
  // Loop over all assemblies
  //   Get the assembly node
  //   Extract the assembly ID from the name
  //   Write the assembly ID to core_assembly_ids
  //   If the assembly ID is not in assembly_ids
  //     Insert the ID to assembly_ids
  //     Get M by N size of the assembly (N = 1 always)
  //     Allocate assembly_lattice_ids to M
  //     Allocate assembly_lattice_zs to M + 1
  //     Loop over all lattices
  //       Get the lattice node
  //       Extract the lattice ID from the name
  //       Write the lattice ID to assembly_lattice_ids
  //       Get the Z positions of the lattice
  //       If this is the first lattice write the top and bottom Z positions to
  //       assembly_lattice_zs Else write the top Z position to assembly_lattice_zs If the
  //       lattice ID is not in lattice_ids
  //         Insert the ID to lattice_ids
  //         Get the M by N size of the lattice
  //         Allocate lattice_rtm_ids to M by N
  //         Loop over all RTMs
  //           Get the RTM node
  //           Extract the RTM ID from the name
  //           Write the RTM ID to lattice_rtm_ids
  //           If the RTM ID is not in rtm_ids
  //             Insert the ID to rtm_ids
  //             Get the M by N size of the RTM
  //             Allocate rtm_coarse_cell_ids to M by N
  //             Loop over all coarse cells
  //               Get the coarse cell node
  //               Extract the coarse cell ID from the name
  //               Write the coarse cell ID to rtm_coarse_cell_ids
  //               If the coarse cell ID is not in coarse_cell_ids
  //                 Insert the ID to coarse_cell_ids
  //                 Read the mesh into a MeshFile object using read_xdmf_uniform_grid
  //                 Set the coarse cell mesh type, mesh id, and material IDs
  //                 Create the mesh
  //                 Use the bounding box of the mesh to set the coarse cell dxdy
  //
  // Now that we have all the IDs we can create the model
  // For each coarse cell,
  //   Use make_coarse_cell to create the coarse cell
  //   Add the mesh to the model
  //   Adjust the mesh id of the coarse cell to be the index of the mesh in the model
  // For each RTM,
  //   Use make_rtm to create the RTM
  // For each lattice,
  //   Use make_lattice to create the lattice
  // For each assembly,
  //   Use make_assembly to create the assembly
  // Use make_core to create the core
  // return the model
  using vint = std::vector<int>;
  using vvint = std::vector<std::vector<int>>;
  using vvvint = std::vector<std::vector<std::vector<int>>>;
  using vdouble = std::vector<double>;
  using vvdouble = std::vector<std::vector<double>>;
  using Vmatid = Vector<MaterialID>;
  using vVmatid = std::vector<Vector<MaterialID>>;
  vvint core_assembly_ids; // 2D map of the IDs of each assembly in the core
  // The assembly IDs are stored in core_assembly_ids[i] as follows:
  // y
  // ^
  // | { { 7, 8, 9},
  // |   { 4, 5, 6}
  // |   { 1, 2, 3} }
  // |
  // +---------> x
  // See the make_core method for more details.
  // This is the same layout for lattices and RTMs
  vvint assembly_lattice_ids;   // 1D map of the IDs of each lattice in each assembly
  vvdouble assembly_lattice_zs; // Z coordinates of each lattice in each assembly
  vint assembly_ids;            // IDs of all assemblies

  vvvint lattice_rtm_ids; // 2D layout of the IDs of each RTM in each lattice
  vint lattice_ids;       // IDs of all lattices

  vvvint rtm_coarse_cell_ids; // 2D layout of the IDs of each coarse cell in each RTM
  vint rtm_ids;               // IDs of all RTMs

  std::vector<Vec2d> coarse_cell_dxdys; // dx and dy of each coarse cell
  vint coarse_cell_mesh_types;          // Mesh type in each coarse cell
  vint coarse_cell_mesh_ids;            // Mesh ID in each coarse cell
  vVmatid coarse_cell_material_ids;     // Material IDs in each coarse cell
  vint coarse_cell_ids;                 // IDs of all coarse cells
  // We also store a Vector of Mesh objects for each mesh type
  Vector<TriMesh<T, I>> tri;
  Vector<QuadMesh<T, I>> quad;
  Vector<TriQuadMesh<T, I>> tri_quad;
  Vector<QuadraticTriMesh<T, I>> quadratic_tri;
  Vector<QuadraticQuadMesh<T, I>> quadratic_quad;
  Vector<QuadraticTriQuadMesh<T, I>> quadratic_tri_quad;

  // Get the core node
  pugi::xml_node xcore = xdomain.child("Grid");
  if (strcmp("Grid", xcore.name()) != 0) {
    Log::error("XDMF XML grid node is not Grid");
    return;
  }
  if (strcmp("Tree", xcore.attribute("GridType").value()) != 0) {
    Log::error("Expected core GridType=Tree");
    return;
  }
  // Get the M by N size of the core (of the form M x N)
  int core_M, core_N;
  {
    pugi::xml_node xcore_mn = xcore.child("Information");
    pugi::xml_attribute xcore_mn_name = xcore_mn.attribute("Name");
    if (strcmp("M_by_N", xcore_mn_name.value()) == 0) {
      std::string xcore_mn_value = xcore_mn.child_value();
      std::stringstream ss(xcore_mn_value);
      std::string token;
      std::getline(ss, token, 'x');
      core_M = std::stoi(token);
      std::getline(ss, token, 'x');
      core_N = std::stoi(token);
    } else {
      Log::error("Expected core Information Name=M_by_N");
      return;
    }
  }
  // Allocate core_assembly_ids to be M by N
  core_assembly_ids.resize(core_M);
  for (int i = 0; i < core_M; i++) {
    core_assembly_ids[i].resize(core_N);
  }
  // Loop over all assemblies
  // Get the assembly node
  int assembly_count = 0;
  for (auto const & assembly_node : xcore.children("Grid")) {
    // Extract the assembly ID from the name
    // Of the form Assembly_XXXXX_YYYYY, where XXXXX is the assembly ID
    std::string assembly_name = assembly_node.attribute("Name").value();
    int assembly_id = std::stoi(assembly_name.substr(9, 5));
    // Write the assembly ID to core_assembly_ids
    {
      int j, i;
      map_lattice_idx_to_j_i(assembly_count, j, i, core_M, core_N);
      core_assembly_ids[j][i] = assembly_id;
    }
    // If the assembly ID is not in assembly_ids
    auto assembly_id_it =
        std::lower_bound(assembly_ids.begin(), assembly_ids.end(), assembly_id);
    if (assembly_id_it == assembly_ids.end() || *assembly_id_it != assembly_id) {
      ptrdiff_t assembly_id_idx = assembly_id_it - assembly_ids.begin();
      // Insert the ID to assembly_ids
      assembly_ids.insert(assembly_id_it, assembly_id);
      // Get M by N size of the assembly (N = 1 always)
      int assembly_M, assembly_N;
      {
        pugi::xml_node xassembly_mn = assembly_node.child("Information");
        pugi::xml_attribute xassembly_mn_name = xassembly_mn.attribute("Name");
        if (strcmp("M_by_N", xassembly_mn_name.value()) == 0) {
          std::string xassembly_mn_value = xassembly_mn.child_value();
          std::stringstream ss(xassembly_mn_value);
          std::string token;
          std::getline(ss, token, 'x');
          assembly_M = std::stoi(token);
          std::getline(ss, token, 'x');
          assembly_N = std::stoi(token);
        } else {
          Log::error("Expected assembly Information Name=M_by_N");
          return;
        }
      }
      if (assembly_N != 1) {
        Log::error("Expected assembly N=1");
        return;
      }
      // Allocate assembly_lattice_ids to M
      assembly_lattice_ids.insert(assembly_lattice_ids.begin() + assembly_id_idx, vint());
      assembly_lattice_ids[assembly_id_idx].resize(assembly_M);
      // Allocate assembly_lattice_zs to M + 1
      assembly_lattice_zs.insert(assembly_lattice_zs.begin() + assembly_id_idx,
                                 vdouble());
      assembly_lattice_zs[assembly_id_idx].resize(assembly_M + 1);
      // Loop over all lattices
      // Get the lattice node
      int lattice_count = 0;
      for (auto const & lattice_node : assembly_node.children("Grid")) {
        // Extract the lattice ID from the name
        // Of the form Lattice_XXXXX_YYYYY, where XXXXX is the lattice ID
        std::string lattice_name = lattice_node.attribute("Name").value();
        int lattice_id = std::stoi(lattice_name.substr(8, 5));
        // Write the lattice ID to assembly_lattice_ids
        assembly_lattice_ids[assembly_id_idx][lattice_count] = lattice_id;
        // Get the Z positions of the lattice
        double lattice_z_low, lattice_z_high;
        {
          pugi::xml_node xlattice_zs = lattice_node.child("Information");
          pugi::xml_attribute xlattice_zs_name = xlattice_zs.attribute("Name");
          if (strcmp("Z", xlattice_zs_name.value()) == 0) {
            std::string xlattice_zs_value = xlattice_zs.child_value();
            std::stringstream ss(xlattice_zs_value);
            std::string token;
            std::getline(ss, token, ',');
            lattice_z_low = std::stod(token);
            std::getline(ss, token, ',');
            std::getline(ss, token, ',');
            lattice_z_high = std::stod(token);
          } else {
            Log::error("Expected assembly Information Name=Z");
            return;
          }
        }
        // If this is the first lattice write the top and bottom Z positions to
        // assembly_lattice_zs Else write the top Z position to assembly_lattice_zs
        if (lattice_count == 0) {
          assembly_lattice_zs[assembly_id_idx][lattice_count] = lattice_z_low;
        }
        assembly_lattice_zs[assembly_id_idx][lattice_count + 1] = lattice_z_high;
        // If the lattice ID is not in lattice_ids
        auto lattice_id_it =
            std::lower_bound(lattice_ids.begin(), lattice_ids.end(), lattice_id);
        if (lattice_id_it == lattice_ids.end() || *lattice_id_it != lattice_id) {
          ptrdiff_t lattice_id_idx = lattice_id_it - lattice_ids.begin();
          // Insert the ID to lattice_ids
          lattice_ids.insert(lattice_ids.begin() + lattice_id_idx, lattice_id);
          // Get M by N size of the lattice
          int lattice_M, lattice_N;
          {
            pugi::xml_node xlattice_mn =
                lattice_node.child("Information").next_sibling("Information");
            pugi::xml_attribute xlattice_mn_name = xlattice_mn.attribute("Name");
            if (strcmp("M_by_N", xlattice_mn_name.value()) == 0) {
              std::string xlattice_mn_value = xlattice_mn.child_value();
              std::stringstream ss(xlattice_mn_value);
              std::string token;
              std::getline(ss, token, 'x');
              lattice_M = std::stoi(token);
              std::getline(ss, token, 'x');
              lattice_N = std::stoi(token);
            } else {
              Log::error("Expected lattice Information Name=M_by_N");
              return;
            }
          }
          // Allocate lattice_rtm_ids to M by N
          lattice_rtm_ids.insert(lattice_rtm_ids.begin() + lattice_id_idx, vvint());
          lattice_rtm_ids[lattice_id_idx].resize(lattice_M);
          for (auto & lattice_rtm_id : lattice_rtm_ids[lattice_id_idx]) {
            lattice_rtm_id.resize(lattice_N);
          }
          // Loop over all RTMs
          // Get the RTM node
          int rtm_count = 0;
          for (auto const & rtm_node : lattice_node.children("Grid")) {
            // Extract the RTM ID from the name
            // Of the form RTM_XXXXX_YYYYY, where XXXXX is the RTM ID
            std::string rtm_name = rtm_node.attribute("Name").value();
            // Write the RTM ID to lattice_rtm_ids
            int rtm_id = std::stoi(rtm_name.substr(5, 5));
            {
              int j, i;
              map_lattice_idx_to_j_i(rtm_count, j, i, lattice_M, lattice_N);
              lattice_rtm_ids[lattice_id_idx][j][i] = rtm_id;
            }
            // If the RTM ID is not in rtm_ids
            auto rtm_id_it = std::lower_bound(rtm_ids.begin(), rtm_ids.end(), rtm_id);
            if (rtm_id_it == rtm_ids.end() || *rtm_id_it != rtm_id) {
              ptrdiff_t rtm_id_idx = rtm_id_it - rtm_ids.begin();
              // Insert the ID to rtm_ids
              rtm_ids.insert(rtm_ids.begin() + rtm_id_idx, rtm_id);
              // Get the M by N size of the RTM
              int rtm_M, rtm_N;
              {
                pugi::xml_node xrtm_mn = rtm_node.child("Information");
                pugi::xml_attribute xrtm_mn_name = xrtm_mn.attribute("Name");
                if (strcmp("M_by_N", xrtm_mn_name.value()) == 0) {
                  std::string xrtm_mn_value = xrtm_mn.child_value();
                  std::stringstream ss(xrtm_mn_value);
                  std::string token;
                  std::getline(ss, token, 'x');
                  rtm_M = std::stoi(token);
                  std::getline(ss, token, 'x');
                  rtm_N = std::stoi(token);
                } else {
                  Log::error("Expected RTM Information Name=M_by_N");
                  return;
                }
              }
              // Allocate rtm_coarse_cell_ids to M by N
              rtm_coarse_cell_ids.insert(rtm_coarse_cell_ids.begin() + rtm_id_idx,
                                         vvint());
              rtm_coarse_cell_ids[rtm_id_idx].resize(rtm_M);
              for (auto & rtm_coarse_cell_id : rtm_coarse_cell_ids[rtm_id_idx]) {
                rtm_coarse_cell_id.resize(rtm_N);
              }
              // Loop over all coarse cells
              // Get the coarse cell node
              int coarse_cell_count = 0;
              for (auto const & coarse_cell_node : rtm_node.children("Grid")) {
                // Extract the coarse cell ID from the name
                // Of the form Coarse_Cell_XXXXX_YYYYY, where XXXXX is the coarse cell ID
                std::string coarse_cell_name = coarse_cell_node.attribute("Name").value();
                // Write the coarse cell ID to rtm_coarse_cell_ids
                int coarse_cell_id = std::stoi(coarse_cell_name.substr(12, 5));
                {
                  int j, i;
                  map_lattice_idx_to_j_i(coarse_cell_count, j, i, rtm_M, rtm_N);
                  rtm_coarse_cell_ids[rtm_id_idx][j][i] = coarse_cell_id;
                }
                // If the coarse cell ID is not in coarse_cell_ids
                auto coarse_cell_id_it = std::lower_bound(
                    coarse_cell_ids.begin(), coarse_cell_ids.end(), coarse_cell_id);
                if (coarse_cell_id_it == coarse_cell_ids.end() ||
                    *coarse_cell_id_it != coarse_cell_id) {
                  ptrdiff_t coarse_cell_id_idx =
                      coarse_cell_id_it - coarse_cell_ids.begin();
                  // Insert the ID to coarse_cell_ids
                  coarse_cell_ids.insert(coarse_cell_ids.begin() + coarse_cell_id_idx,
                                         coarse_cell_id);
                  // Read the mesh into a MeshFile object using read_xdmf_uniform_grid
                  MeshFile<T, I> coarse_cell_mesh;
                  coarse_cell_mesh.format = MeshFileFormat::XDMF;
                  read_xdmf_uniform_grid(coarse_cell_node, h5file, h5filename,
                                         coarse_cell_mesh);
                  // Set the coarse cell mesh type, mesh id, and material IDs
                  MeshType const mesh_type = coarse_cell_mesh.get_mesh_type();
                  coarse_cell_material_ids.insert(
                      coarse_cell_material_ids.begin() + coarse_cell_id_idx, Vmatid());
                  coarse_cell_mesh.get_material_ids(
                      coarse_cell_material_ids[coarse_cell_id_idx], long_material_names);
                  coarse_cell_dxdys.insert(coarse_cell_dxdys.begin() + coarse_cell_id_idx,
                                           Vec2d(0, 0));
                  switch (mesh_type) {
                  case MeshType::TRI: {
                    coarse_cell_mesh_types.insert(coarse_cell_mesh_types.begin() +
                                                      coarse_cell_id_idx,
                                                  static_cast<I>(MeshType::TRI));
                    coarse_cell_mesh_ids.insert(coarse_cell_mesh_ids.begin() +
                                                    coarse_cell_id_idx,
                                                static_cast<I>(tri.size()));
                    tri.push_back(coarse_cell_mesh);
                    // Shift the points so that the min point is at the origin.
                    TriMesh<T, I> & mesh = tri.back();
                    AABox2<T> bb = bounding_box(mesh);
                    Point2<T> min_point = bb.minima;
                    for (auto & p : mesh.vertices) {
                      p -= min_point;
                    }
                    Point2<T> dxdy = bb.maxima - bb.minima;
                    coarse_cell_dxdys[coarse_cell_id_idx] =
                        Vec2d(static_cast<double>(dxdy[0]), static_cast<double>(dxdy[1]));
                    break;
                  }
                  case MeshType::QUAD: {
                    coarse_cell_mesh_types.insert(coarse_cell_mesh_types.begin() +
                                                      coarse_cell_id_idx,
                                                  static_cast<I>(MeshType::QUAD));
                    coarse_cell_mesh_ids.insert(coarse_cell_mesh_ids.begin() +
                                                    coarse_cell_id_idx,
                                                static_cast<I>(quad.size()));
                    quad.push_back(coarse_cell_mesh);
                    // Shift the points so that the min point is at the origin.
                    QuadMesh<T, I> & mesh = quad.back();
                    AABox2<T> bb = bounding_box(mesh);
                    Point2<T> min_point = bb.minima;
                    for (auto & p : mesh.vertices) {
                      p -= min_point;
                    }
                    Point2<T> dxdy = bb.maxima - bb.minima;
                    coarse_cell_dxdys[coarse_cell_id_idx] =
                        Vec2d(static_cast<double>(dxdy[0]), static_cast<double>(dxdy[1]));
                    break;
                  }
                  case MeshType::TRI_QUAD: {
                    coarse_cell_mesh_types.insert(coarse_cell_mesh_types.begin() +
                                                      coarse_cell_id_idx,
                                                  static_cast<I>(MeshType::TRI_QUAD));
                    coarse_cell_mesh_ids.insert(coarse_cell_mesh_ids.begin() +
                                                    coarse_cell_id_idx,
                                                static_cast<I>(tri_quad.size()));
                    tri_quad.push_back(coarse_cell_mesh);
                    // Shift the points so that the min point is at the origin.
                    TriQuadMesh<T, I> & mesh = tri_quad.back();
                    AABox2<T> bb = bounding_box(mesh);
                    Point2<T> min_point = bb.minima;
                    for (auto & p : mesh.vertices) {
                      p -= min_point;
                    }
                    Point2<T> dxdy = bb.maxima - bb.minima;
                    coarse_cell_dxdys[coarse_cell_id_idx] =
                        Vec2d(static_cast<double>(dxdy[0]), static_cast<double>(dxdy[1]));
                    break;
                  }
                  case MeshType::QUADRATIC_TRI: {
                    coarse_cell_mesh_types.insert(
                        coarse_cell_mesh_types.begin() + coarse_cell_id_idx,
                        static_cast<I>(MeshType::QUADRATIC_TRI));
                    coarse_cell_mesh_ids.insert(coarse_cell_mesh_ids.begin() +
                                                    coarse_cell_id_idx,
                                                static_cast<I>(quadratic_tri.size()));
                    quadratic_tri.push_back(coarse_cell_mesh);
                    // Shift the points so that the min point is at the origin.
                    QuadraticTriMesh<T, I> & mesh = quadratic_tri.back();
                    AABox2<T> bb = bounding_box(mesh);
                    Point2<T> min_point = bb.minima;
                    for (auto & p : mesh.vertices) {
                      p -= min_point;
                    }
                    Point2<T> dxdy = bb.maxima - bb.minima;
                    coarse_cell_dxdys[coarse_cell_id_idx] =
                        Vec2d(static_cast<double>(dxdy[0]), static_cast<double>(dxdy[1]));
                    break;
                  }
                  case MeshType::QUADRATIC_QUAD: {
                    coarse_cell_mesh_types.insert(
                        coarse_cell_mesh_types.begin() + coarse_cell_id_idx,
                        static_cast<I>(MeshType::QUADRATIC_QUAD));
                    coarse_cell_mesh_ids.insert(coarse_cell_mesh_ids.begin() +
                                                    coarse_cell_id_idx,
                                                static_cast<I>(quadratic_quad.size()));
                    quadratic_quad.push_back(coarse_cell_mesh);
                    // Shift the points so that the min point is at the origin.
                    QuadraticQuadMesh<T, I> & mesh = quadratic_quad.back();
                    AABox2<T> bb = bounding_box(mesh);
                    Point2<T> min_point = bb.minima;
                    for (auto & p : mesh.vertices) {
                      p -= min_point;
                    }
                    Point2<T> dxdy = bb.maxima - bb.minima;
                    coarse_cell_dxdys[coarse_cell_id_idx] =
                        Vec2d(static_cast<double>(dxdy[0]), static_cast<double>(dxdy[1]));
                    break;
                  }
                  case MeshType::QUADRATIC_TRI_QUAD: {
                    coarse_cell_mesh_types.insert(
                        coarse_cell_mesh_types.begin() + coarse_cell_id_idx,
                        static_cast<I>(MeshType::QUADRATIC_TRI_QUAD));
                    coarse_cell_mesh_ids.insert(
                        coarse_cell_mesh_ids.begin() + coarse_cell_id_idx,
                        static_cast<I>(quadratic_tri_quad.size()));
                    quadratic_tri_quad.push_back(coarse_cell_mesh);
                    // Shift the points so that the min point is at the origin.
                    QuadraticTriQuadMesh<T, I> & mesh = quadratic_tri_quad.back();
                    AABox2<T> bb = bounding_box(mesh);
                    Point2<T> min_point = bb.minima;
                    for (auto & p : mesh.vertices) {
                      p -= min_point;
                    }
                    Point2<T> dxdy = bb.maxima - bb.minima;
                    coarse_cell_dxdys[coarse_cell_id_idx] =
                        Vec2d(static_cast<double>(dxdy[0]), static_cast<double>(dxdy[1]));
                    break;
                  }
                  default:
                    Log::error("Mesh type not supported");
                    return;
                  }
                } // if (coarse_cell_id_it == coarse_cell_ids.end())
                coarse_cell_count++;
              } // for (auto const & coarse_cell_node : rtm_node.children("Grid"))
            }   // if (rtm_id_it == rtm_ids.end())
            rtm_count++;
          } // for (auto const & rtm_node : lattice_node.children("Grid"))
        }   // if (lattice_id_it == lattice_ids.end())
        lattice_count++;
      } // for (auto const & lattice_node : assembly_node.children("Grid"))
    }   // if (assembly_id_it == assembly_ids.end())
    assembly_count++;
  } // for (auto const & assembly_node : xcore.children("Grid"))

  // Now that we have all the IDs we can create the model
  // For each coarse cell,
  for (size_t i = 0; i < coarse_cell_ids.size(); ++i) {
    UM2_ASSERT(static_cast<int>(i) == coarse_cell_ids[i]);
    //   Use make_coarse_cell to create the coarse cell
    model.make_coarse_cell(Vec2<T>(static_cast<T>(coarse_cell_dxdys[i][0]),
                                   static_cast<T>(coarse_cell_dxdys[i][1])));
    // Add the mesh to the model
    // Adjust the mesh id of the coarse cell to be the index of the mesh in the model
    model.coarse_cells[i].mesh_type = coarse_cell_mesh_types[i];
    model.coarse_cells[i].material_ids = coarse_cell_material_ids[i];
    switch (coarse_cell_mesh_types[i]) {
    case static_cast<I>(MeshType::TRI):
      model.coarse_cells[i].mesh_id = model.tri.size();
      model.tri.push_back(tri[coarse_cell_mesh_ids[i]]);
      break;
    case static_cast<I>(MeshType::QUAD):
      model.coarse_cells[i].mesh_id = model.quad.size();
      model.quad.push_back(quad[coarse_cell_mesh_ids[i]]);
      break;
    case static_cast<I>(MeshType::TRI_QUAD):
      model.coarse_cells[i].mesh_id = model.tri_quad.size();
      model.tri_quad.push_back(tri_quad[coarse_cell_mesh_ids[i]]);
      break;
    case static_cast<I>(MeshType::QUADRATIC_TRI):
      model.coarse_cells[i].mesh_id = model.quadratic_tri.size();
      model.quadratic_tri.push_back(quadratic_tri[coarse_cell_mesh_ids[i]]);
      break;
    case static_cast<I>(MeshType::QUADRATIC_QUAD):
      model.coarse_cells[i].mesh_id = model.quadratic_quad.size();
      model.quadratic_quad.push_back(quadratic_quad[coarse_cell_mesh_ids[i]]);
      break;
    case static_cast<I>(MeshType::QUADRATIC_TRI_QUAD):
      model.coarse_cells[i].mesh_id = model.quadratic_tri_quad.size();
      model.quadratic_tri_quad.push_back(quadratic_tri_quad[coarse_cell_mesh_ids[i]]);
      break;
    default:
      Log::error("Mesh type not supported");
      return;
    }
  }
  // For each RTM,
  for (size_t i = 0; i < rtm_ids.size(); ++i) {
    UM2_ASSERT(static_cast<int>(i) == rtm_ids[i]);
    model.make_rtm(rtm_coarse_cell_ids[i]);
  }
  // For each lattice,
  for (size_t i = 0; i < lattice_ids.size(); ++i) {
    UM2_ASSERT(static_cast<int>(i) == lattice_ids[i]);
    model.make_lattice(lattice_rtm_ids[i]);
  }
  // For each assembly,
  for (size_t i = 0; i < assembly_ids.size(); ++i) {
    UM2_ASSERT(static_cast<int>(i) == assembly_ids[i]);
    UM2_ASSERT(
        std::is_sorted(assembly_lattice_zs[i].begin(), assembly_lattice_zs[i].end()));
    model.make_assembly(assembly_lattice_ids[i], assembly_lattice_zs[i]);
  }
  model.make_core(core_assembly_ids);
  return;
}

} // namespace um2
