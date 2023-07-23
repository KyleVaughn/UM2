#pragma once

#include <um2/config.hpp>

#include <um2/common/Color.hpp>
#include <um2/common/String.hpp>
#include <um2/mesh/RegularPartition.hpp>

#include <cstdlib> // exit
#include <cstdio> // fprintf

namespace um2
{

// An image can be represented as a 2D array of pixels. However, since we only wish to use
// this class for rendering objects in the 2D plane (while preserving the relative shapes and 
// positions of objects), we can use a RegularPartition<2, T, Color> to represent the image.
//
// This has the advantage of allowing us to map multiple geometric objects to the same image
// without having to worry about scaling, computing the correct pixel coordinates, etc.
//
// NOTE: We must apply the constraint that spacing[0] == spacing[1] to ensure that the image
//      is not distorted.
template <typename T>
struct Image2D : public RegularPartition<2, T, Color> {

  constexpr Image2D() noexcept = default;

  // ----------------------------------------------------------------------------
  // Methods
  // ----------------------------------------------------------------------------

  void write(String const & filename) const;

  template <uint64_t N>
  void write(char const (&filename)[N]) const;
  
};

} // namespace um2

#include "Image2D.inl"
