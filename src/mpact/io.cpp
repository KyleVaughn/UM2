#include <um2/mpact/io.hpp>

namespace um2
{

//==============================================================================
// writeCoarseCell
//==============================================================================

static void
writeCoarseCell(Size rtm_id, mpact::SpatialPartition const & model, Size ix, Size iy,
                Vector<Int> & cc_found, Point2<Float> const & prev_ll,
                std::vector<std::string> const & mat_names, Float const cut_z,
                std::stringstream & ss, pugi::xml_node & xrtm_grid, H5::H5File & h5file,
                std::string const & h5filename, std::string const & h5rtm_grouppath,
                std::vector<std::string> const & mat_names_short)
{
  auto const & rtm = model.rtms[rtm_id];
  auto const & cell_id = static_cast<Size>(rtm.getChild(ix, iy));
  cc_found[cell_id] += 1;
  auto const cell_bb = rtm.getBox(ix, iy);
  Point2<Float> const ll = prev_ll + cell_bb.minima;
  Int const cell_id_ctr = cc_found[cell_id];
  MeshType const mesh_type = model.coarse_cells[cell_id].mesh_type;
  Size const mesh_id = model.coarse_cells[cell_id].mesh_id;
  MeshFile<Float, Int> mesh_file;
  switch (mesh_type) {
  case MeshType::Tri:
    model.tri[mesh_id].toMeshFile(mesh_file);
    break;
  case MeshType::Quad:
    model.quad[mesh_id].toMeshFile(mesh_file);
    break;
  case MeshType::QuadraticTri:
    model.quadratic_tri[mesh_id].toMeshFile(mesh_file);
    break;
  case MeshType::QuadraticQuad:
    model.quadratic_quad[mesh_id].toMeshFile(mesh_file);
    break;
  default:
    Log::error("Unsupported mesh type");
    return;
  } // switch
  // We need to add the material_ids as elsets to the mesh file
  // Get the number of materials in the mesh
  Vector<MaterialID> const & mat_ids = model.coarse_cells[cell_id].material_ids;
  Vector<MaterialID> unique_mat_ids;
  for (auto const & mat_id : mat_ids) {
    if (!std::any_of(unique_mat_ids.begin(), unique_mat_ids.end(),
                     [mat_id](MaterialID const & id) { return id == mat_id; })) {
      unique_mat_ids.push_back(mat_id);
    }
  }
  auto const num_mats = static_cast<size_t>(unique_mat_ids.size());
  std::sort(unique_mat_ids.begin(), unique_mat_ids.end());
  mesh_file.elset_names.resize(num_mats);
  mesh_file.elset_offsets.resize(num_mats + 1U);
  mesh_file.elset_offsets[0] = 0;
  size_t const num_elements = mesh_file.numCells();
  size_t elem_ctr = 0;
  mesh_file.elset_ids.resize(num_elements);
  // Indexing with different types makes the next few lines a bit messy
  for (size_t imat = 0; imat < num_mats; ++imat) {
    MaterialID const mat_id = unique_mat_ids[static_cast<Size>(imat)];
    mesh_file.elset_names[imat] = mat_names[static_cast<size_t>(mat_id)];
    for (size_t ielem = 0; ielem < num_elements; ++ielem) {
      if (mat_ids[static_cast<Size>(ielem)] == mat_id) {
        mesh_file.elset_ids[elem_ctr++] = static_cast<Size>(ielem);
        mesh_file.elset_offsets[imat + 1U] = static_cast<Size>(elem_ctr);
      }
    }
  }
  // Shift the mesh to global coordinates
  for (auto & vertex : mesh_file.vertices) {
    vertex[0] += ll[0];
    vertex[1] += ll[1];
    vertex[2] += cut_z;
  }
  ss.str("");
  ss << "Coarse_Cell_" << std::setw(5) << std::setfill('0') << cell_id << "_"
     << std::setw(5) << std::setfill('0') << cell_id_ctr;
  mesh_file.name = ss.str();
  // Now write the coarse cell as a uniform grid
  writeXDMFUniformGrid(xrtm_grid, h5file, h5filename, h5rtm_grouppath, mesh_file,
                       mat_names_short);
}

//==============================================================================
// writeRTM
//==============================================================================

static void
writeRTM(Size lat_id, mpact::SpatialPartition const & model, Size ix, Size iy,
         Vector<Int> & cc_found, Vector<Int> & rtm_found, Point2<Float> const & asy_ll,
         std::vector<std::string> const & mat_names, Float const cut_z,
         std::stringstream & ss, pugi::xml_node & xlat_grid, H5::H5File & h5file,
         std::string const & h5filename, std::string const & h5lat_grouppath,
         std::vector<std::string> const & mat_names_short)
{
  auto const & lattice = model.lattices[lat_id];
  auto const rtm_id = static_cast<Size>(lattice.getChild(ix, iy));
  rtm_found[rtm_id] += 1;
  Int const rtm_id_ctr = rtm_found[rtm_id];
  pugi::xml_node xrtm_grid = xlat_grid.append_child("Grid");
  ss.str("");
  ss << "RTM_" << std::setw(5) << std::setfill('0') << rtm_id << "_" << std::setw(5)
     << std::setfill('0') << rtm_id_ctr;
  xrtm_grid.append_attribute("Name") = ss.str().c_str();
  xrtm_grid.append_attribute("GridType") = "Tree";
  std::string const h5rtm_grouppath = h5lat_grouppath + "/" + ss.str();
  H5::Group const h5rtm_group = h5file.createGroup(h5rtm_grouppath);
  auto const rtm_bb = lattice.getBox(ix, iy);
  Point2<Float> const rtm_ll = rtm_bb.minima; // Lower left corner
  auto const & rtm = model.rtms[rtm_id];
  if (rtm.children.empty()) {
    Log::error("RTM has no children");
    return;
  }
  Size const nycells = rtm.numYCells();
  Size const nxcells = rtm.numXCells();
  // RTM M by N
  pugi::xml_node xrtm_info = xrtm_grid.append_child("Information");
  xrtm_info.append_attribute("Name") = "M_by_N";
  std::string const rtm_mn_str =
      std::to_string(nycells) + " x " + std::to_string(nxcells);
  xrtm_info.append_child(pugi::node_pcdata).set_value(rtm_mn_str.c_str());
  Point2<Float> const prev_ll = asy_ll + rtm_ll;
  // For each coarse cell
  for (Size iycell = 0; iycell < nycells; ++iycell) {
    for (Size ixcell = 0; ixcell < nxcells; ++ixcell) {
      writeCoarseCell(rtm_id, model, ixcell, iycell, cc_found, prev_ll, mat_names, cut_z,
                      ss, xrtm_grid, h5file, h5filename, h5rtm_grouppath,
                      mat_names_short);
    } // cell
  }   // cell
}

//==============================================================================
// writeLattice
//==============================================================================

static void
writeLattice(Size asy_id, mpact::SpatialPartition const & model, Size iz,
             Vector<Int> & cc_found, Vector<Int> & rtm_found, Vector<Int> & lat_found,
             Point2<Float> const & asy_ll, std::vector<std::string> const & mat_names,
             std::stringstream & ss, pugi::xml_node & xasy_grid, H5::H5File & h5file,
             std::string const & h5filename, std::string const & h5asy_grouppath,
             std::vector<std::string> const & mat_names_short)
{
  auto const & assembly = model.assemblies[asy_id];
  auto const lat_id = static_cast<Size>(assembly.children[iz]);
  lat_found[lat_id] += 1;
  Int const lat_id_ctr = lat_found[lat_id];
  pugi::xml_node xlat_grid = xasy_grid.append_child("Grid");
  ss.str("");
  ss << "Lattice_" << std::setw(5) << std::setfill('0') << lat_id << "_" << std::setw(5)
     << std::setfill('0') << lat_id_ctr;
  xlat_grid.append_attribute("Name") = ss.str().c_str();
  xlat_grid.append_attribute("GridType") = "Tree";
  std::string const h5lat_grouppath = h5asy_grouppath + "/" + ss.str();
  H5::Group const h5lat_group = h5file.createGroup(h5lat_grouppath);
  Float const low_z = assembly.grid.divs[0][iz];
  Float const high_z = assembly.grid.divs[0][iz + 1];
  Float const cut_z = (low_z + high_z) / 2;
  pugi::xml_node xlat_info = xlat_grid.append_child("Information");
  xlat_info.append_attribute("Name") = "Z";
  std::string const z_values = std::to_string(low_z) + ", " + std::to_string(cut_z) +
                               ", " + std::to_string(high_z);
  xlat_info.append_child(pugi::node_pcdata).set_value(z_values.c_str());
  auto const & lattice = model.lattices[lat_id];
  if (lattice.children.empty()) {
    Log::error("Lattice has no children");
    return;
  }
  Size const nyrtm = lattice.numYCells();
  Size const nxrtm = lattice.numXCells();
  // Lattice M by N
  pugi::xml_node xlat_info2 = xlat_grid.append_child("Information");
  xlat_info2.append_attribute("Name") = "M_by_N";
  std::string const lat_mn_str = std::to_string(nyrtm) + " x " + std::to_string(nxrtm);
  xlat_info2.append_child(pugi::node_pcdata).set_value(lat_mn_str.c_str());
  // For each RTM
  for (Size iyrtm = 0; iyrtm < nyrtm; ++iyrtm) {
    for (Size ixrtm = 0; ixrtm < nxrtm; ++ixrtm) {
      writeRTM(lat_id, model, ixrtm, iyrtm, cc_found, rtm_found, asy_ll, mat_names, cut_z,
               ss, xlat_grid, h5file, h5filename, h5lat_grouppath, mat_names_short);
    } // rtm
  }   // rtm
}

//==============================================================================
// writeXDMFFile
//==============================================================================

void
writeXDMFFile(std::string const & path, mpact::SpatialPartition const & model)
{
  Log::info("Writing MPACT model to XDMF file: " + path);

  size_t const h5filepath_end = path.find_last_of('/') + 1;
  std::string const name = path.substr(h5filepath_end, path.size() - 5 - h5filepath_end);
  std::string const h5filename = name + ".h5";
  std::string const h5filepath = path.substr(0, h5filepath_end);
  LOG_DEBUG("H5 filename: " + h5filename);
  H5::H5File h5file(h5filepath + h5filename, H5F_ACC_TRUNC);

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
  std::string materials;
  std::vector<std::string> mat_names;
  std::vector<std::string> mat_names_short;
  std::string const material_str = "Material_";
  for (Size i = 0; i < model.materials.size(); i++) {
    std::string const mat_name(model.materials[i].name.data());
    materials += mat_name;
    if (i + 1 < model.materials.size()) {
      materials += ", ";
    }
    mat_names.push_back(material_str + mat_name);
    mat_names_short.push_back(mat_name);
  }
  xinfo.append_child(pugi::node_pcdata).set_value(materials.c_str());

  // Core grid
  pugi::xml_node xcore_grid = xdomain.append_child("Grid");
  xcore_grid.append_attribute("Name") = name.c_str();
  xcore_grid.append_attribute("GridType") = "Tree";

  // h5
  H5::Group const h5core_group = h5file.createGroup(name);
  std::string const h5core_grouppath = "/" + name;

  Vector<Int> cc_found(model.coarse_cells.size(), -1);
  Vector<Int> rtm_found(model.rtms.size(), -1);
  Vector<Int> lat_found(model.lattices.size(), -1);
  Vector<Int> asy_found(model.assemblies.size(), -1);
  auto const & core = model.core;
  if (core.children.empty()) {
    Log::error("Core has no children");
    return;
  }
  std::stringstream ss;
  Size const nyasy = core.numYCells();
  Size const nxasy = core.numXCells();
  // Core M by N
  pugi::xml_node xcore_info = xcore_grid.append_child("Information");
  xcore_info.append_attribute("Name") = "M_by_N";
  std::string const core_mn_str = std::to_string(nyasy) + " x " + std::to_string(nxasy);
  xcore_info.append_child(pugi::node_pcdata).set_value(core_mn_str.c_str());
  // For each assembly
  for (Size iyasy = 0; iyasy < nyasy; ++iyasy) {
    for (Size ixasy = 0; ixasy < nxasy; ++ixasy) {
      Int const asy_id = core.getChild(ixasy, iyasy);
      asy_found[static_cast<Size>(asy_id)] += 1;
      Int const asy_id_ctr = asy_found[static_cast<Size>(asy_id)];
      pugi::xml_node xasy_grid = xcore_grid.append_child("Grid");
      ss.str("");
      ss << "Assembly_" << std::setw(5) << std::setfill('0') << asy_id << "_"
         << std::setw(5) << std::setfill('0') << asy_id_ctr;
      xasy_grid.append_attribute("Name") = ss.str().c_str();
      xasy_grid.append_attribute("GridType") = "Tree";
      std::string const h5asy_grouppath = h5core_grouppath + "/" + ss.str();
      H5::Group const h5asy_group = h5file.createGroup(h5asy_grouppath);
      AxisAlignedBox2<Float> const asy_bb = core.getBox(ixasy, iyasy);
      Point2<Float> const asy_ll = asy_bb.minima; // Lower left corner
      auto const & assembly = model.assemblies[asy_id];
      if (assembly.children.empty()) {
        Log::error("Assembly has no children");
        return;
      }
      Size const nzlat = assembly.numXCells();
      // Assembly M by N
      pugi::xml_node xasy_info = xasy_grid.append_child("Information");
      xasy_info.append_attribute("Name") = "M_by_N";
      std::string const asy_mn_str = std::to_string(nzlat) + " x 1";
      xasy_info.append_child(pugi::node_pcdata).set_value(asy_mn_str.c_str());
      // For each lattice
      for (Size izlat = 0; izlat < nzlat; ++izlat) {
        writeLattice(asy_id, model, izlat, cc_found, rtm_found, lat_found, asy_ll,
                     mat_names, ss, xasy_grid, h5file, h5filename, h5asy_grouppath,
                     mat_names_short);
      } // lat
    }   // assembly
  }     // assembly
  // Write the XML file
  xdoc.save_file(path.c_str(), "  ");

  // Close the HDF5 file
  h5file.close();
}

//==============================================================================
// exportMesh
//==============================================================================

void
exportMesh(std::string const & path, mpact::SpatialPartition const & model)
{
  if (path.ends_with(".xdmf")) {
    writeXDMFFile(path, model);
  } else {
    Log::error("Unsupported file format.");
  }
}

//==============================================================================
// mapLatticeIndexToji
//==============================================================================

static inline void
mapLatticeIndexToji(int const idx, int & j, int & i, int const m, int const n)
{
  j = m - idx / n - 1;
  i = idx % n;
}

//==============================================================================
// readXDMFFile
//==============================================================================

void
// NOLINTNEXTLINE
readXDMFFile(std::string const & path, mpact::SpatialPartition & model)
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
  size_t const h5filepath_end = path.find_last_of('/') + 1;
  std::string const h5filename =
      path.substr(h5filepath_end, path.size() - 4 - h5filepath_end) + "h5";
  std::string const h5filepath = path.substr(0, h5filepath_end);
  LOG_DEBUG("H5 filename: " + h5filename);
  H5::H5File const h5file(h5filepath + h5filename, H5F_ACC_RDONLY);

  // Setup XML document
  pugi::xml_document xdoc;
  pugi::xml_parse_result const result = xdoc.load_file(path.c_str());
  if (!result) {
    Log::error("XDMF XML parse error: " + std::string(result.description()) +
               ", character pos= " + std::to_string(result.offset));
  }
  pugi::xml_node const xroot = xdoc.child("Xdmf");
  if (strcmp("Xdmf", xroot.name()) != 0) {
    Log::error("XDMF XML root node is not Xdmf");
    return;
  }
  pugi::xml_node const xdomain = xroot.child("Domain");
  if (strcmp("Domain", xdomain.name()) != 0) {
    Log::error("XDMF XML domain node is not Domain");
    return;
  }
  std::vector<std::string> material_names;
  pugi::xml_node const xinfo = xdomain.child("Information");
  if (strcmp("Information", xinfo.name()) == 0) {
    // Get the "Name" attribute
    pugi::xml_attribute const xname = xinfo.attribute("Name");
    if (strcmp("Materials", xname.value()) == 0) {
      // Get the material names
      std::string const materials = xinfo.child_value();
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
  model.materials.resize(static_cast<Size>(material_names.size()));
  for (size_t imat = 0; imat < material_names.size(); ++imat) {
    model.materials[static_cast<Size>(imat)].name =
        ShortString(material_names[imat].c_str());
  }
  std::vector<std::string> long_material_names(material_names.size());
  for (size_t imat = 0; imat < material_names.size(); ++imat) {
    long_material_names[imat] = "Material_" + material_names[imat];
  }

  //============================================================================
  // Algorithm for populating the model
  //============================================================================
  //
  // Constraints:
  // - We want to use the SpatialPartition.makeCore, makeAssembly, makeLattice,
  // makeRTM, and makeCoarseCell methods to create the model. These functions take
  // children IDs as arguments.
  // - We have to create ID 1 before ID 2, ID 2 before ID 3, etc. since the make*
  // methods do not take its own ID as an argument. Therefore we have to construct
  // the model in a bottom-up fashion, i.e. we have to create the coarse cells before
  // we can create the RTMs, etc.
  // - We want to avoid making multiple passes over the XDMF file.
  //
  // Algorithm:
  // ========================================================================
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
  //       assembly_lattice_zs Else write the top Z position to assembly_lattice_zs
  //       If the lattice ID is not in lattice_ids
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
  //                 Read the mesh into a MeshFile object using readXDMFUniformGrid
  //                 Set the coarse cell mesh type, mesh id, and material IDs
  //                 Create the mesh
  //                 Use the bounding box of the mesh to set the coarse cell dxdy
  //
  // Now that we have all the IDs we can create the model
  // For each coarse cell,
  //   Use makeCoarseCell to create the coarse cell
  //   Add the mesh to the model
  //   Adjust the mesh id of the coarse cell to be the index of the mesh in the model
  // For each RTM,
  //   Use makeRTM to create the RTM
  // For each lattice,
  //   Use makeLattice to create the lattice
  // For each assembly,
  //   Use makeAssembly to create the assembly
  // Use makeCore to create the core
  // return the model

  // NOLINTBEGIN(readability-identifier-naming) justification: STL types
  using vint = std::vector<int>;
  using vvint = std::vector<std::vector<int>>;
  using vvvint = std::vector<std::vector<std::vector<int>>>;
  using vdouble = std::vector<double>;
  using vvdouble = std::vector<std::vector<double>>;
  // using Vmatid = Vector<MaterialID>;
  // using vVmatid = std::vector<Vector<MaterialID>>;
  //  NOLINTEND(readability-identifier-naming)

  vvint core_assembly_ids; // 2D map of the IDs of each assembly in the core
  // The assembly IDs are stored in core_assembly_ids[i] as follows:
  // y
  // ^
  // | { { 7, 8, 9},
  // |   { 4, 5, 6}
  // |   { 1, 2, 3} }
  // |
  // +---------> x
  // See the makeCore method for more details.
  // This is the same layout for lattices and RTMs
  vvint assembly_lattice_ids;   // 1D map of the IDs of each lattice in each assembly
  vvdouble assembly_lattice_zs; // Z coordinates of each lattice in each assembly
  vint assembly_ids;            // IDs of all assemblies

  vvvint lattice_rtm_ids; // 2D layout of the IDs of each RTM in each lattice
  vint lattice_ids;       // IDs of all lattices

  // vvvint rtm_coarse_cell_ids; // 2D layout of the IDs of each coarse cell in each RTM
  vint rtm_ids; // IDs of all RTMs
  //
  //   std::vector<Vec2d> coarse_cell_dxdys; // dx and dy of each coarse cell
  //   vint coarse_cell_mesh_types;          // Mesh type in each coarse cell
  //   vint coarse_cell_mesh_ids;            // Mesh ID in each coarse cell
  //   vVmatid coarse_cell_material_ids;     // Material IDs in each coarse cell
  //   vint coarse_cell_ids;                 // IDs of all coarse cells
  //   // We also store a Vector of Mesh objects for each mesh type
  //   Vector<TriMesh<T, I>> tri;
  //   Vector<QuadMesh<T, I>> quad;
  //   Vector<TriQuadMesh<T, I>> tri_quad;
  //   Vector<QuadraticTriMesh<T, I>> quadratic_tri;
  //   Vector<QuadraticQuadMesh<T, I>> quadratic_quad;
  //   Vector<QuadraticTriQuadMesh<T, I>> quadratic_tri_quad;
  //
  // Get the core node
  pugi::xml_node const xcore = xdomain.child("Grid");
  if (strcmp("Grid", xcore.name()) != 0) {
    Log::error("XDMF XML grid node is not Grid");
    return;
  }
  if (strcmp("Tree", xcore.attribute("GridType").value()) != 0) {
    Log::error("Expected core GridType=Tree");
    return;
  }
  // Get the M by N size of the core (of the form M x N)
  int core_m = 0;
  int core_n = 0;
  {
    pugi::xml_node const xcore_mn = xcore.child("Information");
    pugi::xml_attribute const xcore_mn_name = xcore_mn.attribute("Name");
    if (strcmp("M_by_N", xcore_mn_name.value()) == 0) {
      std::string const xcore_mn_value = xcore_mn.child_value();
      std::stringstream ss(xcore_mn_value);
      std::string token;
      std::getline(ss, token, 'x');
      core_m = sto<int>(token);
      std::getline(ss, token, 'x');
      core_n = sto<int>(token);
      assert(core_m > 0);
      assert(core_n > 0);
    } else {
      Log::error("Expected core Information Name=M_by_N");
      return;
    }
  }
  // Allocate core_assembly_ids to be M by N
  core_assembly_ids.resize(static_cast<size_t>(core_m));
  for (size_t i = 0; i < static_cast<size_t>(core_m); i++) {
    core_assembly_ids[i].resize(static_cast<size_t>(core_n));
  }
  // Loop over all assemblies
  // Get the assembly node
  int assembly_count = 0;
  for (auto const & assembly_node : xcore.children("Grid")) {
    // Extract the assembly ID from the name
    // Of the form Assembly_XXXXX_YYYYY, where XXXXX is the assembly ID
    std::string const assembly_name = assembly_node.attribute("Name").value();
    int const assembly_id = sto<int>(assembly_name.substr(9, 5));
    // Write the assembly ID to core_assembly_ids
    {
      int j = 0;
      int i = 0;
      mapLatticeIndexToji(assembly_count, j, i, core_m, core_n);
      assert(j >= 0 && j < core_m);
      assert(i >= 0 && i < core_n);
      core_assembly_ids[static_cast<size_t>(j)][static_cast<size_t>(i)] = assembly_id;
    }
    // If the assembly ID is not in assembly_ids
    auto assembly_id_it =
        std::lower_bound(assembly_ids.begin(), assembly_ids.end(), assembly_id);
    if (assembly_id_it == assembly_ids.end() || *assembly_id_it != assembly_id) {
      ptrdiff_t const assembly_id_idx = assembly_id_it - assembly_ids.begin();
      // Insert the ID to assembly_ids
      assembly_ids.insert(assembly_id_it, assembly_id);
      // Get M by N size of the assembly (N = 1 always)
      int assembly_m = 0;
      int assembly_n = 0;
      {
        pugi::xml_node const xassembly_mn = assembly_node.child("Information");
        pugi::xml_attribute const xassembly_mn_name = xassembly_mn.attribute("Name");
        if (strcmp("M_by_N", xassembly_mn_name.value()) == 0) {
          std::string const xassembly_mn_value = xassembly_mn.child_value();
          std::stringstream ss(xassembly_mn_value);
          std::string token;
          std::getline(ss, token, 'x');
          assembly_m = sto<int>(token);
          std::getline(ss, token, 'x');
          assembly_n = sto<int>(token);
          assert(assembly_m > 0);
          assert(assembly_n > 0);
        } else {
          Log::error("Expected assembly Information Name=M_by_N");
          return;
        }
      }
      if (assembly_n != 1) {
        Log::error("Expected assembly N=1");
        return;
      }
      // Allocate assembly_lattice_ids to M
      assembly_lattice_ids.insert(assembly_lattice_ids.begin() + assembly_id_idx, vint());
      assembly_lattice_ids[static_cast<size_t>(assembly_id_idx)].resize(
          static_cast<size_t>(assembly_m));
      // Allocate assembly_lattice_zs to M + 1
      assembly_lattice_zs.insert(assembly_lattice_zs.begin() + assembly_id_idx,
                                 vdouble());
      assembly_lattice_zs[static_cast<size_t>(assembly_id_idx)].resize(
          static_cast<size_t>(assembly_m) + 1U);
      // Loop over all lattices
      // Get the lattice node
      int lattice_count = 0;
      for (auto const & lattice_node : assembly_node.children("Grid")) {
        // Extract the lattice ID from the name
        // Of the form Lattice_XXXXX_YYYYY, where XXXXX is the lattice ID
        std::string const lattice_name = lattice_node.attribute("Name").value();
        int const lattice_id = sto<int>(lattice_name.substr(8, 5));
        // Write the lattice ID to assembly_lattice_ids
        assembly_lattice_ids[static_cast<size_t>(assembly_id_idx)]
                            [static_cast<size_t>(lattice_count)] = lattice_id;
        // Get the Z positions of the lattice
        double lattice_z_low = 1e10;
        double lattice_z_high = -1e10;
        {
          pugi::xml_node const xlattice_zs = lattice_node.child("Information");
          pugi::xml_attribute const xlattice_zs_name = xlattice_zs.attribute("Name");
          if (strcmp("Z", xlattice_zs_name.value()) == 0) {
            std::string const xlattice_zs_value = xlattice_zs.child_value();
            std::stringstream ss(xlattice_zs_value);
            std::string token;
            std::getline(ss, token, ',');
            lattice_z_low = sto<double>(token);
            std::getline(ss, token, ',');
            std::getline(ss, token, ',');
            lattice_z_high = sto<double>(token);
          } else {
            Log::error("Expected assembly Information Name=Z");
            return;
          }
          assert(lattice_z_low < lattice_z_high);
          assert(lattice_z_low < 1e9);
          assert(lattice_z_high > -1e9);
        }
        // If this is the first lattice write the top and bottom Z positions to
        // assembly_lattice_zs else write the top Z position to assembly_lattice_zs
        if (lattice_count == 0) {
          assembly_lattice_zs[static_cast<size_t>(assembly_id_idx)]
                             [static_cast<size_t>(lattice_count)] = lattice_z_low;
        }
        assembly_lattice_zs[static_cast<size_t>(assembly_id_idx)]
                           [static_cast<size_t>(lattice_count) + 1U] = lattice_z_high;
        // If the lattice ID is not in lattice_ids
        auto lattice_id_it =
            std::lower_bound(lattice_ids.begin(), lattice_ids.end(), lattice_id);
        if (lattice_id_it == lattice_ids.end() || *lattice_id_it != lattice_id) {
          ptrdiff_t const lattice_id_idx = lattice_id_it - lattice_ids.begin();
          // Insert the ID to lattice_ids
          lattice_ids.insert(lattice_ids.begin() + lattice_id_idx, lattice_id);
          // Get M by N size of the lattice
          int lattice_m = 0;
          int lattice_n = 0;
          {
            pugi::xml_node const xlattice_mn =
                lattice_node.child("Information").next_sibling("Information");
            pugi::xml_attribute const xlattice_mn_name = xlattice_mn.attribute("Name");
            if (strcmp("M_by_N", xlattice_mn_name.value()) == 0) {
              std::string const xlattice_mn_value = xlattice_mn.child_value();
              std::stringstream ss(xlattice_mn_value);
              std::string token;
              std::getline(ss, token, 'x');
              lattice_m = sto<int>(token);
              std::getline(ss, token, 'x');
              lattice_n = sto<int>(token);
              assert(lattice_m > 0);
              assert(lattice_n > 0);
            } else {
              Log::error("Expected lattice Information Name=M_by_N");
              return;
            }
          }
          // Allocate lattice_rtm_ids to M by N
          lattice_rtm_ids.insert(lattice_rtm_ids.begin() + lattice_id_idx, vvint());
          lattice_rtm_ids[static_cast<size_t>(lattice_id_idx)].resize(
              static_cast<size_t>(lattice_m));
          for (auto & lattice_rtm_id :
               lattice_rtm_ids[static_cast<size_t>(lattice_id_idx)]) {
            lattice_rtm_id.resize(static_cast<size_t>(lattice_n));
          }
          // Loop over all RTMs
          // Get the RTM node
          int rtm_count = 0;
          for (auto const & rtm_node : lattice_node.children("Grid")) {
            // Extract the RTM ID from the name
            // Of the form RTM_XXXXX_YYYYY, where XXXXX is the RTM ID
            std::string const rtm_name = rtm_node.attribute("Name").value();
            // Write the RTM ID to lattice_rtm_ids
            int const rtm_id = sto<int>(rtm_name.substr(5, 5));
            {
              int j = 0;
              int i = 0;
              mapLatticeIndexToji(rtm_count, j, i, lattice_m, lattice_n);
              lattice_rtm_ids[static_cast<size_t>(lattice_id_idx)][static_cast<size_t>(j)]
                             [static_cast<size_t>(i)] = rtm_id;
            }
            // If the RTM ID is not in rtm_ids
            auto rtm_id_it = std::lower_bound(rtm_ids.begin(), rtm_ids.end(), rtm_id);
            if (rtm_id_it == rtm_ids.end() || *rtm_id_it != rtm_id) {
              ptrdiff_t const rtm_id_idx = rtm_id_it - rtm_ids.begin();
              // Insert the ID to rtm_ids
              rtm_ids.insert(rtm_ids.begin() + rtm_id_idx, rtm_id);
              // Get the M by N size of the RTM
              int rtm_m = 0;
              int rtm_n = 0;
              {
                pugi::xml_node const xrtm_mn = rtm_node.child("Information");
                pugi::xml_attribute const xrtm_mn_name = xrtm_mn.attribute("Name");
                if (strcmp("M_by_N", xrtm_mn_name.value()) == 0) {
                  std::string const xrtm_mn_value = xrtm_mn.child_value();
                  std::stringstream ss(xrtm_mn_value);
                  std::string token;
                  std::getline(ss, token, 'x');
                  rtm_m = sto<int>(token);
                  std::getline(ss, token, 'x');
                  rtm_n = sto<int>(token);
                  assert(rtm_m > 0);
                  assert(rtm_n > 0);
                } else {
                  Log::error("Expected RTM Information Name=M_by_N");
                  return;
                }
              }
              //               // Allocate rtm_coarse_cell_ids to M by N
              //               rtm_coarse_cell_ids.insert(rtm_coarse_cell_ids.begin() +
              //               rtm_id_idx,
              //                                          vvint());
              //               rtm_coarse_cell_ids[rtm_id_idx].resize(rtm_M);
              //               for (auto & rtm_coarse_cell_id :
              //               rtm_coarse_cell_ids[rtm_id_idx])
              //               {
              //                 rtm_coarse_cell_id.resize(rtm_N);
              //               }
              //               // Loop over all coarse cells
              //               // Get the coarse cell node
              //               int coarse_cell_count = 0;
              //               for (auto const & coarse_cell_node :
              //               rtm_node.children("Grid"))
              //               {
              //                 // Extract the coarse cell ID from the name
              //                 // Of the form Coarse_Cell_XXXXX_YYYYY, where XXXXX is
              //                 the coarse cell ID std::string coarse_cell_name =
              //                 coarse_cell_node.attribute("Name").value();
              //                 // Write the coarse cell ID to rtm_coarse_cell_ids
              //                 int coarse_cell_id =
              //                 std::stoi(coarse_cell_name.substr(12, 5));
              //                 {
              //                   int j, i;
              //                   map_lattice_idx_to_j_i(coarse_cell_count, j, i, rtm_M,
              //                   rtm_N); rtm_coarse_cell_ids[rtm_id_idx][j][i] =
              //                   coarse_cell_id;
              //                 }
              //                 // If the coarse cell ID is not in coarse_cell_ids
              //                 auto coarse_cell_id_it = std::lower_bound(
              //                     coarse_cell_ids.begin(), coarse_cell_ids.end(),
              //                     coarse_cell_id);
              //                 if (coarse_cell_id_it == coarse_cell_ids.end() ||
              //                     *coarse_cell_id_it != coarse_cell_id) {
              //                   ptrdiff_t coarse_cell_id_idx =
              //                       coarse_cell_id_it - coarse_cell_ids.begin();
              //                   // Insert the ID to coarse_cell_ids
              //                   coarse_cell_ids.insert(coarse_cell_ids.begin() +
              //                   coarse_cell_id_idx,
              //                                          coarse_cell_id);
              //                   // Read the mesh into a MeshFile object using
              //                   read_xdmf_uniform_grid MeshFile<T, I> coarse_cell_mesh;
              //                   coarse_cell_mesh.format = MeshFileFormat::XDMF;
              //                   read_xdmf_uniform_grid(coarse_cell_node, h5file,
              //                   h5filename,
              //                                          coarse_cell_mesh);
              //                   // Set the coarse cell mesh type, mesh id, and material
              //                   IDs MeshType const mesh_type =
              //                   coarse_cell_mesh.get_mesh_type();
              //                   coarse_cell_material_ids.insert(
              //                       coarse_cell_material_ids.begin() +
              //                       coarse_cell_id_idx, Vmatid());
              //                   coarse_cell_mesh.get_material_ids(
              //                       coarse_cell_material_ids[coarse_cell_id_idx],
              //                       long_material_names);
              //                   coarse_cell_dxdys.insert(coarse_cell_dxdys.begin() +
              //                   coarse_cell_id_idx,
              //                                            Vec2d(0, 0));
              //                   switch (mesh_type) {
              //                   case MeshType::TRI: {
              //                     coarse_cell_mesh_types.insert(coarse_cell_mesh_types.begin()
              //                     +
              //                                                       coarse_cell_id_idx,
              //                                                   static_cast<Int>(MeshType::TRI));
              //                     coarse_cell_mesh_ids.insert(coarse_cell_mesh_ids.begin()
              //                     +
              //                                                     coarse_cell_id_idx,
              //                                                 static_cast<Int>(tri.size()));
              //                     tri.push_back(coarse_cell_mesh);
              //                     // Shift the points so that the min point is at the
              //                     origin. TriMesh<T, I> & mesh = tri.back();
              //                     AxisAlignedBox2<Float> bb = bounding_box(mesh);
              //                     Point2<Float> min_point = bb.minima;
              //                     for (auto & p : mesh.vertices) {
              //                       p -= min_point;
              //                     }
              //                     Point2<Float> dxdy = bb.maxima - bb.minima;
              //                     coarse_cell_dxdys[coarse_cell_id_idx] =
              //                         Vec2d(static_cast<double>(dxdy[0]),
              //                         static_cast<double>(dxdy[1]));
              //                     break;
              //                   }
              //                   case MeshType::QUAD: {
              //                     coarse_cell_mesh_types.insert(coarse_cell_mesh_types.begin()
              //                     +
              //                                                       coarse_cell_id_idx,
              //                                                   static_cast<Int>(MeshType::QUAD));
              //                     coarse_cell_mesh_ids.insert(coarse_cell_mesh_ids.begin()
              //                     +
              //                                                     coarse_cell_id_idx,
              //                                                 static_cast<Int>(quad.size()));
              //                     quad.push_back(coarse_cell_mesh);
              //                     // Shift the points so that the min point is at the
              //                     origin. QuadMesh<T, I> & mesh = quad.back();
              //                     AxisAlignedBox2<Float> bb = bounding_box(mesh);
              //                     Point2<Float> min_point = bb.minima;
              //                     for (auto & p : mesh.vertices) {
              //                       p -= min_point;
              //                     }
              //                     Point2<Float> dxdy = bb.maxima - bb.minima;
              //                     coarse_cell_dxdys[coarse_cell_id_idx] =
              //                         Vec2d(static_cast<double>(dxdy[0]),
              //                         static_cast<double>(dxdy[1]));
              //                     break;
              //                   }
              //                   case MeshType::TRI_QUAD: {
              //                     coarse_cell_mesh_types.insert(coarse_cell_mesh_types.begin()
              //                     +
              //                                                       coarse_cell_id_idx,
              //                                                   static_cast<Int>(MeshType::TRI_QUAD));
              //                     coarse_cell_mesh_ids.insert(coarse_cell_mesh_ids.begin()
              //                     +
              //                                                     coarse_cell_id_idx,
              //                                                 static_cast<Int>(tri_quad.size()));
              //                     tri_quad.push_back(coarse_cell_mesh);
              //                     // Shift the points so that the min point is at the
              //                     origin. TriQuadMesh<T, I> & mesh = tri_quad.back();
              //                     AxisAlignedBox2<Float> bb = bounding_box(mesh);
              //                     Point2<Float> min_point = bb.minima;
              //                     for (auto & p : mesh.vertices) {
              //                       p -= min_point;
              //                     }
              //                     Point2<Float> dxdy = bb.maxima - bb.minima;
              //                     coarse_cell_dxdys[coarse_cell_id_idx] =
              //                         Vec2d(static_cast<double>(dxdy[0]),
              //                         static_cast<double>(dxdy[1]));
              //                     break;
              //                   }
              //                   case MeshType::QUADRATIC_TRI: {
              //                     coarse_cell_mesh_types.insert(
              //                         coarse_cell_mesh_types.begin() +
              //                         coarse_cell_id_idx,
              //                         static_cast<Int>(MeshType::QUADRATIC_TRI));
              //                     coarse_cell_mesh_ids.insert(coarse_cell_mesh_ids.begin()
              //                     +
              //                                                     coarse_cell_id_idx,
              //                                                 static_cast<Int>(quadratic_tri.size()));
              //                     quadratic_tri.push_back(coarse_cell_mesh);
              //                     // Shift the points so that the min point is at the
              //                     origin. QuadraticTriMesh<T, I> & mesh =
              //                     quadratic_tri.back(); AxisAlignedBox2<Float> bb =
              //                     bounding_box(mesh); Point2<Float> min_point =
              //                     bb.minima; for (auto & p : mesh.vertices) {
              //                       p -= min_point;
              //                     }
              //                     Point2<Float> dxdy = bb.maxima - bb.minima;
              //                     coarse_cell_dxdys[coarse_cell_id_idx] =
              //                         Vec2d(static_cast<double>(dxdy[0]),
              //                         static_cast<double>(dxdy[1]));
              //                     break;
              //                   }
              //                   case MeshType::QUADRATIC_QUAD: {
              //                     coarse_cell_mesh_types.insert(
              //                         coarse_cell_mesh_types.begin() +
              //                         coarse_cell_id_idx,
              //                         static_cast<Int>(MeshType::QUADRATIC_QUAD));
              //                     coarse_cell_mesh_ids.insert(coarse_cell_mesh_ids.begin()
              //                     +
              //                                                     coarse_cell_id_idx,
              //                                                 static_cast<Int>(quadratic_quad.size()));
              //                     quadratic_quad.push_back(coarse_cell_mesh);
              //                     // Shift the points so that the min point is at the
              //                     origin. QuadraticQuadMesh<T, I> & mesh =
              //                     quadratic_quad.back(); AxisAlignedBox2<Float> bb =
              //                     bounding_box(mesh); Point2<Float> min_point =
              //                     bb.minima; for (auto & p : mesh.vertices) {
              //                       p -= min_point;
              //                     }
              //                     Point2<Float> dxdy = bb.maxima - bb.minima;
              //                     coarse_cell_dxdys[coarse_cell_id_idx] =
              //                         Vec2d(static_cast<double>(dxdy[0]),
              //                         static_cast<double>(dxdy[1]));
              //                     break;
              //                   }
              //                   case MeshType::QUADRATIC_TRI_QUAD: {
              //                     coarse_cell_mesh_types.insert(
              //                         coarse_cell_mesh_types.begin() +
              //                         coarse_cell_id_idx,
              //                         static_cast<Int>(MeshType::QUADRATIC_TRI_QUAD));
              //                     coarse_cell_mesh_ids.insert(
              //                         coarse_cell_mesh_ids.begin() +
              //                         coarse_cell_id_idx,
              //                         static_cast<Int>(quadratic_tri_quad.size()));
              //                     quadratic_tri_quad.push_back(coarse_cell_mesh);
              //                     // Shift the points so that the min point is at the
              //                     origin. QuadraticTriQuadMesh<T, I> & mesh =
              //                     quadratic_tri_quad.back(); AxisAlignedBox2<Float> bb
              //                     = bounding_box(mesh); Point2<Float> min_point =
              //                     bb.minima; for (auto & p : mesh.vertices) {
              //                       p -= min_point;
              //                     }
              //                     Point2<Float> dxdy = bb.maxima - bb.minima;
              //                     coarse_cell_dxdys[coarse_cell_id_idx] =
              //                         Vec2d(static_cast<double>(dxdy[0]),
              //                         static_cast<double>(dxdy[1]));
              //                     break;
              //                   }
              //                   default:
              //                     Log::error("Mesh type not supported");
              //                     return;
              //                   }
              //                 } // if (coarse_cell_id_it == coarse_cell_ids.end())
              //                 coarse_cell_count++;
              //               } // for (auto const & coarse_cell_node :
              //               rtm_node.children("Grid"))
            } // if (rtm_id_it == rtm_ids.end())
            rtm_count++;
          } // for (auto const & rtm_node : lattice_node.children("Grid"))
        }   // if (lattice_id_it == lattice_ids.end())
        lattice_count++;
      } // for (auto const & lattice_node : assembly_node.children("Grid"))
    }   // if (assembly_id_it == assembly_ids.end())
    assembly_count++;
  } // for (auto const & assembly_node : xcore.children("Grid"))
  //
  //   // Now that we have all the IDs we can create the model
  //   // For each coarse cell,
  //   for (size_t i = 0; i < coarse_cell_ids.size(); ++i) {
  //     UM2_ASSERT(static_cast<int>(i) == coarse_cell_ids[i]);
  //     //   Use makeCoarseCell to create the coarse cell
  //     model.makeCoarseCell(Vec2<Float>(static_cast<Float>(coarse_cell_dxdys[i][0]),
  //                                    static_cast<Float>(coarse_cell_dxdys[i][1])));
  //     // Add the mesh to the model
  //     // Adjust the mesh id of the coarse cell to be the index of the mesh in the
  //     model model.coarse_cells[i].mesh_type = coarse_cell_mesh_types[i];
  //     model.coarse_cells[i].material_ids = coarse_cell_material_ids[i];
  //     switch (coarse_cell_mesh_types[i]) {
  //     case static_cast<Int>(MeshType::TRI):
  //       model.coarse_cells[i].mesh_id = model.tri.size();
  //       model.tri.push_back(tri[coarse_cell_mesh_ids[i]]);
  //       break;
  //     case static_cast<Int>(MeshType::QUAD):
  //       model.coarse_cells[i].mesh_id = model.quad.size();
  //       model.quad.push_back(quad[coarse_cell_mesh_ids[i]]);
  //       break;
  //     case static_cast<Int>(MeshType::TRI_QUAD):
  //       model.coarse_cells[i].mesh_id = model.tri_quad.size();
  //       model.tri_quad.push_back(tri_quad[coarse_cell_mesh_ids[i]]);
  //       break;
  //     case static_cast<Int>(MeshType::QUADRATIC_TRI):
  //       model.coarse_cells[i].mesh_id = model.quadratic_tri.size();
  //       model.quadratic_tri.push_back(quadratic_tri[coarse_cell_mesh_ids[i]]);
  //       break;
  //     case static_cast<Int>(MeshType::QUADRATIC_QUAD):
  //       model.coarse_cells[i].mesh_id = model.quadratic_quad.size();
  //       model.quadratic_quad.push_back(quadratic_quad[coarse_cell_mesh_ids[i]]);
  //       break;
  //     case static_cast<Int>(MeshType::QUADRATIC_TRI_QUAD):
  //       model.coarse_cells[i].mesh_id = model.quadratic_tri_quad.size();
  //       model.quadratic_tri_quad.push_back(quadratic_tri_quad[coarse_cell_mesh_ids[i]]);
  //       break;
  //     default:
  //       Log::error("Mesh type not supported");
  //       return;
  //     }
  //   }
  //   // For each RTM,
  //   for (size_t i = 0; i < rtm_ids.size(); ++i) {
  //     UM2_ASSERT(static_cast<int>(i) == rtm_ids[i]);
  //     model.makeRTM(rtm_coarse_cell_ids[i]);
  //   }
  //   // For each lattice,
  //   for (size_t i = 0; i < lattice_ids.size(); ++i) {
  //     UM2_ASSERT(static_cast<int>(i) == lattice_ids[i]);
  //     model.makeLattice(lattice_rtm_ids[i]);
  //   }
  //   // For each assembly,
  //   for (size_t i = 0; i < assembly_ids.size(); ++i) {
  //     UM2_ASSERT(static_cast<int>(i) == assembly_ids[i]);
  //     UM2_ASSERT(
  //         std::is_sorted(assembly_lattice_zs[i].begin(),
  //         assembly_lattice_zs[i].end()));
  //     model.makeAssembly(assembly_lattice_ids[i], assembly_lattice_zs[i]);
  //   }
  //   model.makeCore(core_assembly_ids);
}

void
importMesh(std::string const & path, mpact::SpatialPartition & model)
{
  if (path.ends_with(".xdmf")) {
    readXDMFFile(path, model);
  } else {
    Log::error("Unsupported file format.");
  }
}

} // namespace um2
