#include <bit>

namespace um2
{

// -----------------------------------------------------------------------------
// Constructors
// -----------------------------------------------------------------------------

UM2_HOSTDEV constexpr Color::Color() noexcept : r(0), g(0), b(0), a(255) {}

template <std::integral I>
UM2_HOSTDEV constexpr Color::Color(I r_in, I g_in, I b_in, I a_in) noexcept
    : r(static_cast<uint8_t>(r_in)), g(static_cast<uint8_t>(g_in)),
      b(static_cast<uint8_t>(b_in)), a(static_cast<uint8_t>(a_in))
{
}

template <std::floating_point T>
UM2_HOSTDEV constexpr Color::Color(T r_in, T g_in, T b_in, T a_in) noexcept
    : r(static_cast<uint8_t>(r_in * 255)), g(static_cast<uint8_t>(g_in * 255)),
      b(static_cast<uint8_t>(b_in * 255)), a(static_cast<uint8_t>(a_in * 255))
{
}

template <size_t N>
Color::Color(char const (&name)[N]) noexcept : Color(String(name))
{
}

// -----------------------------------------------------------------------------
// Operators
// -----------------------------------------------------------------------------

#ifdef __CUDA_ARCH__
UM2_CONST __device__ constexpr auto operator==(Color const lhs, Color const rhs) noexcept
    -> bool
{
  return reinterpret_cast<uint32_t const &>(lhs) ==
         reinterpret_cast<uint32_t const &>(rhs);
}

UM2_CONST __device__ constexpr auto operator<(Color const lhs, Color const rhs) noexcept
    -> bool
{
  return reinterpret_cast<uint32_t const &>(lhs) <
         reinterpret_cast<uint32_t const &>(rhs);
}
#else
UM2_CONST UM2_HOST constexpr auto operator==(Color const lhs, Color const rhs) noexcept
    -> bool
{
  return std::bit_cast<uint32_t>(lhs) == std::bit_cast<uint32_t>(rhs);
}
UM2_CONST UM2_HOST constexpr auto operator<(Color const lhs, Color const rhs) noexcept
    -> bool
{
  return std::bit_cast<uint32_t>(lhs) < std::bit_cast<uint32_t>(rhs);
}
#endif

UM2_CONST UM2_HOSTDEV constexpr auto operator!=(Color const lhs, Color const rhs) noexcept
    -> bool
{
  return !(lhs == rhs);
}

} // namespace um2
