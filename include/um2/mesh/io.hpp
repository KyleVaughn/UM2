#pragma once

#include <um2/mesh/io_abaqus.hpp>
#include <um2/mesh/io_xdmf.hpp>

namespace um2
{

// -----------------------------------------------------------------------------
// Mesh file
// -----------------------------------------------------------------------------
// IO for mesh files.

template <std::floating_point T, std::signed_integral I>
void
importMesh(std::string const & path, MeshFile<T, I> & mesh)
{
  if (path.ends_with(".inp")) {
    readAbaqusFile<T, I>(path, mesh);
  } else if (path.ends_with(".xdmf")) {
    readXDMFFile<T, I>(path, mesh);
  } else {
    Log::error("Unsupported file format.");
  }
}

template <std::floating_point T, std::signed_integral I>
void
exportMesh(std::string const & path, MeshFile<T, I> & mesh)
{
  if (path.ends_with(".xdmf")) {
    mesh.filepath = path;
    writeXDMFFile<T, I>(mesh);
  } else {
    Log::error("Unsupported file format.");
  }
}

} // namespace um2
