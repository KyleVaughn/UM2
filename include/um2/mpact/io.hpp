#pragma once

#include <um2/mesh/io_xdmf.hpp>
#include <um2/mpact/SpatialPartition.hpp>

namespace um2
{

void
writeXDMFFile(std::string const & path, mpact::SpatialPartition const & model);

void
exportMesh(std::string const & path, mpact::SpatialPartition const & model);

void
importMesh(std::string const & path, mpact::SpatialPartition & model);

} // namespace um2
