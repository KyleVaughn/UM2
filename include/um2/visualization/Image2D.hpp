#pragma once

#include <um2/config.hpp>

#include <um2/common/Color.hpp>
#include <um2/common/Log.hpp>
#include <um2/geometry/Dion.hpp>
#include <um2/mesh/RegularPartition.hpp>

#include <cstdlib> // exit
#include <string>  // string

namespace um2
{

// An image can be represented as a 2D array of pixels. However, since we only wish to use
// this class for rendering objects in the 2D plane (while preserving the relative shapes
// and positions of objects), we can use a RegularPartition<2, T, Color> to represent the
// image.
//
// This has the advantage of allowing us to map multiple geometric objects to the same
// image without having to worry about scaling, computing the correct pixel coordinates,
// etc.
//
// However, the memory layout of images is typically row-major, top-to-bottom. This class
// uses row-major, bottom-to-top indexing to be consistent with the coordinate system of
// the 2D plane.
//
// NOTE: We must apply the constraint that spacing[0] == spacing[1] to ensure that the
// image is not distorted.
template <std::floating_point T>
struct Image2D : public RegularPartition<2, T, Color> {

  //============================================================================
  // Default point rasterization parameters
  //============================================================================

  static constexpr T default_point_radius = static_cast<T>(0.05);
  static constexpr Colors default_point_color = Colors::White;

  //============================================================================
  // Default line rasterization parameters
  //============================================================================

  static constexpr T default_line_thickness = static_cast<T>(0.01);
  static constexpr Colors default_line_color = Colors::White;

  constexpr Image2D() noexcept = default;

  //============================================================================
  // Methods
  //============================================================================

  void
  clear(Color c = Colors::Black);

  void
  write(std::string const & filename) const;

  void
  rasterize(Point2<T> const & p, Color c = default_point_color);

  void
  rasterizeAsDisk(Point2<T> const & p, T r = default_point_radius,
                  Color c = default_point_color);

  void
  rasterize(LineSegment2<T> const & l, Color c = default_line_color);
};

} // namespace um2

#include "Image2D.inl"
