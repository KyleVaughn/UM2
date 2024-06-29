#include <um2.hpp>

// NOLINTBEGIN(misc-include-cleaner, readability-function-cognitive-complexity)

auto
main(int argc, char ** argv) -> int
{
  um2::initialize();

  // Get the file name from the command line
  if (argc != 2) {
    um2::String const exec_name(argv[0]);
    um2::logger::error("Usage: ", exec_name, " <filename>");
    return 1;
  }

  // Read the MPACT model
  um2::String const filename(argv[1]);
  um2::mpact::Model model;
  model.read(filename);

  // Store the global mean chord lengths in a vector
  um2::Vector<Float> global_mcls(model.numFineCellsTotal());

  // Store the mean chord lengths of the fine cells of each coarse cell
  // to avoid recomputing them
  Int const num_coarse_cells = model.numCoarseCells();
  um2::Vector<um2::Vector<Float>> coarse_cell_mcls(num_coarse_cells);

  // Loop over the coarse cells, computing the mean chord lengths of the fine cells
  for (Int icc = 0; icc < num_coarse_cells; ++icc) {
    auto const & cc = model.getCoarseCell(icc);
    auto & cc_mcls = coarse_cell_mcls[icc];
    cc_mcls.resize(cc.numFaces());
    switch (cc.mesh_type) {
    case um2::MeshType::Tri: {
      auto const & mesh = model.getTriMesh(cc.mesh_id);
      for (Int iface = 0; iface < mesh.numFaces(); ++iface) {
        cc_mcls[iface] = mesh.getFace(iface).meanChordLength();
      }
    } break;
    case um2::MeshType::Quad: {
      auto const & mesh = model.getQuadMesh(cc.mesh_id);
      for (Int iface = 0; iface < mesh.numFaces(); ++iface) {
        cc_mcls[iface] = mesh.getFace(iface).meanChordLength();
      }
    } break;
    case um2::MeshType::QuadraticTri: {
      auto const & mesh = model.getTri6Mesh(cc.mesh_id);
      for (Int iface = 0; iface < mesh.numFaces(); ++iface) {
        cc_mcls[iface] = mesh.getFace(iface).meanChordLength();
      }
    } break;
    case um2::MeshType::QuadraticQuad: {
      auto const & mesh = model.getQuad8Mesh(cc.mesh_id);
      for (Int iface = 0; iface < mesh.numFaces(); ++iface) {
        cc_mcls[iface] = mesh.getFace(iface).meanChordLength();
      }
    } break;
    default:
      LOG_ERROR("Invalid mesh type");
      return 1;
    }
  } // for (Int i = 0; i < num_coarse_cells; ++i)

  // Loop over the model hierarchy, writing the mean chord lengths of the fine cells
  // to the global vector
  Int cell_count = 0;
  for (auto const & asy_id : model.core().children()) {
    for (auto const & lat_id : model.assemblies()[asy_id].children()) {
      for (auto const & rtm_id : model.lattices()[lat_id].children()) {
        for (auto const & cc_id : model.rtms()[rtm_id].children()) {
          Int const cc_num_faces = coarse_cell_mcls[cc_id].size();
          um2::copy(coarse_cell_mcls[cc_id].begin(), coarse_cell_mcls[cc_id].end(),
                    global_mcls.begin() + cell_count);
          cell_count += cc_num_faces;
        }
      }
    }
  }

  // Write the global mean chord lengths to a file
  um2::String const out_filename("mcls.txt");
  FILE * file = fopen(out_filename.data(), "w");
  if (file == nullptr) {
    LOG_ERROR("Failed to open file: ", out_filename);
    return 1;
  }

  // Print with 16 decimal places
  for (auto const mcl : global_mcls) {
    int const ret = fprintf(file, "%.16f\n", mcl);
    if (ret < 0) {
      LOG_ERROR("Failed to write to file: ", out_filename);
      return 1;
    }
  }

  int const ret = fclose(file);
  if (ret != 0) {
    LOG_ERROR("Failed to close file: ", out_filename);
    return 1;
  }

  um2::finalize();
  return 0;
}

// NOLINTEND(misc-include-cleaner, readability-function-cognitive-complexity)
