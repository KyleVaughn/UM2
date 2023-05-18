#include <bit>

namespace um2
{

// Constructors
// -----------------------------------------------------------------------------

UM2_HOSTDEV constexpr Color::Color() : a(255) {}

template <std::integral I>
UM2_HOSTDEV constexpr Color::Color(I r, I g, I b, I a)
    : r(static_cast<uint8_t>(r)), g(static_cast<uint8_t>(g)), b(static_cast<uint8_t>(b)),
      a(static_cast<uint8_t>(a))
{
}

template <std::floating_point T>
UM2_HOSTDEV constexpr Color::Color(T r, T g, T b, T a)
    : r(static_cast<uint8_t>(r * 255)), g(static_cast<uint8_t>(g * 255)),
      b(static_cast<uint8_t>(b * 255)), a(static_cast<uint8_t>(a * 255))
{
}

// Operators
// -----------------------------------------------------------------------------

UM2_CONST UM2_HOSTDEV constexpr auto operator==(Color const lhs, Color const rhs) -> bool
{
  return std::bit_cast<uint32_t>(lhs) == std::bit_cast<uint32_t>(rhs);
}

UM2_CONST UM2_HOSTDEV constexpr auto operator!=(Color const lhs, Color const rhs) -> bool
{
  return !(lhs == rhs);
}

UM2_CONST UM2_HOSTDEV constexpr auto operator<(Color const lhs, Color const rhs) -> bool
{
  return std::bit_cast<uint32_t>(lhs) < std::bit_cast<uint32_t>(rhs);
}

} // namespace um2