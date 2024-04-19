#include <um2.hpp>

#include <iostream>

auto
// NOLINTNEXTLINE(readability-function-cognitive-complexity)
main(int argc, char **argv) -> int 
{
  um2::initialize();

  // Get the file name from the command line
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <filename>" << std::endl;
    return 1;
  }

  // Read the MPACT model
  um2::String const filename(argv[1]);
  um2::mpact::Model model;
  model.read(filename);

  // Store the global edge lengths in a vector
  um2::Vector<Float> global_edge_lens;
  global_edge_lens.reserve(4 * model.numFineCellsTotal());

  // Store the edge lengths of the fine cells of each coarse cell
  // to avoid recomputing them
  Int const num_coarse_cells = model.numCoarseCells();
  um2::Vector<um2::Vector<Float>> coarse_cell_edge_lens(num_coarse_cells);

  // Loop over the coarse cells, computing the edge lengths of the fine cells
  for (Int icc = 0; icc < num_coarse_cells; ++icc) {
    auto const & cc = model.getCoarseCell(icc);
    auto & cc_edge_lens = coarse_cell_edge_lens[icc];
    switch (cc.mesh_type) {
    case um2::MeshType::Tri:
      {
        Int const num_edges_per_face = 3;
        cc_edge_lens.resize(num_edges_per_face * cc.numFaces());
        auto const & mesh = model.getTriMesh(cc.mesh_id);
        for (Int iface = 0; iface < mesh.numFaces(); ++iface) {
          for (Int iedge = 0; iedge < num_edges_per_face; ++iedge) {
            cc_edge_lens[iface * num_edges_per_face + iedge] = 
              mesh.getEdge(iface, iedge).length();
          }
        }
      }
      break;
    case um2::MeshType::Quad:
      {
        Int const num_edges_per_face = 4;
        cc_edge_lens.resize(num_edges_per_face * cc.numFaces());
        auto const & mesh = model.getQuadMesh(cc.mesh_id);
        for (Int iface = 0; iface < mesh.numFaces(); ++iface) {
          for (Int iedge = 0; iedge < num_edges_per_face; ++iedge) {
            cc_edge_lens[iface * num_edges_per_face + iedge] = 
              mesh.getEdge(iface, iedge).length();
          }
        }
      }
      break;
    case um2::MeshType::QuadraticTri:
      {
        Int const num_edges_per_face = 3;
        cc_edge_lens.resize(num_edges_per_face * cc.numFaces());
        auto const & mesh = model.getTri6Mesh(cc.mesh_id);
        for (Int iface = 0; iface < mesh.numFaces(); ++iface) {
          for (Int iedge = 0; iedge < num_edges_per_face; ++iedge) {
            cc_edge_lens[iface * num_edges_per_face + iedge] = 
              mesh.getEdge(iface, iedge).length();
          }
        }
      }
      break;
    case um2::MeshType::QuadraticQuad:
      {
        Int const num_edges_per_face = 4;
        cc_edge_lens.resize(num_edges_per_face * cc.numFaces());
        auto const & mesh = model.getQuad8Mesh(cc.mesh_id);
        for (Int iface = 0; iface < mesh.numFaces(); ++iface) {
          for (Int iedge = 0; iedge < num_edges_per_face; ++iedge) {
            cc_edge_lens[iface * num_edges_per_face + iedge] = 
              mesh.getEdge(iface, iedge).length();
          }
        }
      }
      break;
    default:
      LOG_ERROR("Invalid mesh type");
      return 1;
    }
  } // for (Int i = 0; i < num_coarse_cells; ++i)

  // Loop over the model hierarchy, writing the edge lengths of the fine cells
  // to the global vector
  for (auto const & asy_id : model.core().children()) {
    for (auto const & lat_id : model.assemblies()[asy_id].children()) {
      for (auto const & rtm_id : model.lattices()[lat_id].children()) {
        for (auto const & cc_id : model.rtms()[rtm_id].children()) {
          for (auto const len : coarse_cell_edge_lens[cc_id]) {
            global_edge_lens.emplace_back(len);
          }
        }
      }
    }
  }

  // Write the global edge lengths to a file
  um2::String const out_filename("edge_lens.txt");
  FILE * file = fopen(out_filename.data(), "w");
  if (file == nullptr) {
    LOG_ERROR("Failed to open file: ", out_filename);
    return 1;
  }

  // Print with 16 decimal places
  for (auto const edge_len : global_edge_lens) {
    int const ret = fprintf(file, "%.16f\n", edge_len);
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
