#pragma once

#include <um2/mesh/io_abaqus.hpp>
#include <um2/mesh/io_xdmf.hpp>

namespace um2
{

template <std::floating_point T, std::signed_integral I>
void
importMesh(String const & path, PolytopeSoup<T, I> & mesh)
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
exportMesh(String const & path, PolytopeSoup<T, I> const & mesh)
{
  if (path.ends_with(".xdmf")) {
    writeXDMFFile<T, I>(path, mesh);
  } else {
    Log::error("Unsupported file format.");
  }
}

} // namespace um2
