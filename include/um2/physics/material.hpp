#pragma once

#include <um2/common/color.hpp>
#include <um2/common/string.hpp>
#include <um2/physics/cross_section.hpp>

namespace um2
{

template <std::floating_point T>
class Material {

  String _name;
  Color _color;
  CrossSection<T> _xs;

  public:

  //======================================================================
  // Constructors
  //======================================================================

  constexpr Material() noexcept = default;

  HOSTDEV constexpr Material(String const & name, Color color) noexcept
      : _name(name),
        _color(color)
  {
  }

  // TODO(kcvaughn): implicitly convert for name and color

  //  HOSTDEV constexpr Material(ShortString const & name_in,
  //                             ShortString const & color_in) noexcept
  //      : name(name_in),
  //        color(color_in)
  //  {
  //  }
  //
  //  template <uint64_t M, uint64_t N>
  //  HOSTDEV constexpr Material(char const (&name_in)[M], char const (&color_in)[N])
  //  noexcept
  //      : name(name_in),
  //        color(color_in)
  //  {
  //  }
};

} // namespace um2
