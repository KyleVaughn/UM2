#include <um2/math/morton.hpp>

#include "../test_macros.hpp"

template <std::unsigned_integral U>
HOSTDEV
TEST_CASE(mortonEncode)
{
  // NOLINTNEXTLINE justification: reduce clutter
  using namespace um2;

  // 2D
  ASSERT(mortonEncode(static_cast<U>(0), static_cast<U>(0)) == 0);
  ASSERT(mortonEncode(static_cast<U>(1), static_cast<U>(0)) == 1);
  ASSERT(mortonEncode(static_cast<U>(0), static_cast<U>(1)) == 2);
  ASSERT(mortonEncode(static_cast<U>(1), static_cast<U>(1)) == 3);
  ASSERT(mortonEncode(static_cast<U>(2), static_cast<U>(2)) == 12);
  ASSERT(mortonEncode(static_cast<U>(3), static_cast<U>(3)) == 15);

  // 3D
  ASSERT(mortonEncode(static_cast<U>(0), static_cast<U>(0), static_cast<U>(0)) == 0);
  ASSERT(mortonEncode(static_cast<U>(1), static_cast<U>(0), static_cast<U>(0)) == 1);
  ASSERT(mortonEncode(static_cast<U>(0), static_cast<U>(1), static_cast<U>(0)) == 2);
  ASSERT(mortonEncode(static_cast<U>(1), static_cast<U>(1), static_cast<U>(0)) == 3);
  ASSERT(mortonEncode(static_cast<U>(0), static_cast<U>(0), static_cast<U>(1)) == 4);
  ASSERT(mortonEncode(static_cast<U>(1), static_cast<U>(0), static_cast<U>(1)) == 5);
  ASSERT(mortonEncode(static_cast<U>(0), static_cast<U>(1), static_cast<U>(1)) == 6);
  ASSERT(mortonEncode(static_cast<U>(1), static_cast<U>(1), static_cast<U>(1)) == 7);
  ASSERT(mortonEncode(static_cast<U>(2), static_cast<U>(0), static_cast<U>(0)) == 8);
  ASSERT(mortonEncode(static_cast<U>(2), static_cast<U>(2), static_cast<U>(2)) == 56);
  ASSERT(mortonEncode(static_cast<U>(3), static_cast<U>(3), static_cast<U>(3)) == 63);
}

template <std::unsigned_integral U>
HOSTDEV
TEST_CASE(mortonDecode)
{

  // NOLINTNEXTLINE justification: reduce clutter
  using namespace um2;

  // 2D
  U x;
  U y;
  mortonDecode(static_cast<U>(0), x, y);
  ASSERT(x == 0 && y == 0);
  mortonDecode(static_cast<U>(1), x, y);
  ASSERT(x == 1 && y == 0);
  mortonDecode(static_cast<U>(2), x, y);
  ASSERT(x == 0 && y == 1);
  mortonDecode(static_cast<U>(3), x, y);
  ASSERT(x == 1 && y == 1);
  mortonDecode(static_cast<U>(12), x, y);
  ASSERT(x == 2 && y == 2);
  mortonDecode(static_cast<U>(15), x, y);
  ASSERT(x == 3 && y == 3);

  // 3D
  U z;
  mortonDecode(static_cast<U>(0), x, y, z);
  ASSERT(x == 0 && y == 0 && z == 0);
  mortonDecode(static_cast<U>(1), x, y, z);
  ASSERT(x == 1 && y == 0 && z == 0);
  mortonDecode(static_cast<U>(2), x, y, z);
  ASSERT(x == 0 && y == 1 && z == 0);
  mortonDecode(static_cast<U>(3), x, y, z);
  ASSERT(x == 1 && y == 1 && z == 0);
  mortonDecode(static_cast<U>(4), x, y, z);
  ASSERT(x == 0 && y == 0 && z == 1);
  mortonDecode(static_cast<U>(5), x, y, z);
  ASSERT(x == 1 && y == 0 && z == 1);
  mortonDecode(static_cast<U>(6), x, y, z);
  ASSERT(x == 0 && y == 1 && z == 1);
  mortonDecode(static_cast<U>(7), x, y, z);
  ASSERT(x == 1 && y == 1 && z == 1);
  mortonDecode(static_cast<U>(56), x, y, z);
  ASSERT(x == 2 && y == 2 && z == 2);
}

template <std::unsigned_integral U, std::floating_point T>
HOSTDEV
TEST_CASE(mortonEncodeFloat)
{
  // NOLINTNEXTLINE justification: reduce clutter
  using namespace um2;

  T const zero = static_cast<T>(0);
  T const one = static_cast<T>(1);
  ASSERT((mortonEncode<U, T>(zero, zero) == static_cast<U>(0x0000000000000000)));
  ASSERT((mortonEncode<U, T>(one, zero) == static_cast<U>(0x5555555555555555)));
  ASSERT((mortonEncode<U, T>(zero, one) == static_cast<U>(0xAAAAAAAAAAAAAAAA)));
  ASSERT((mortonEncode<U, T>(one, one) == static_cast<U>(0xFFFFFFFFFFFFFFFF)));

  // 3D
  U value = 100;
  ASSERT((mortonEncode<U, T>(zero, zero, zero) == static_cast<U>(0x0000000000000000)));
  value = mortonEncode<U, T>(one, one, one);
  if constexpr (std::same_as<uint32_t, U>) {
    ASSERT(value == static_cast<U>(0x3fffffff));
  } else {
    ASSERT(value == static_cast<U>(0x7fffffffffffffff));
  }
  value = mortonEncode<U, T>(one, zero, zero);
  if constexpr (std::same_as<uint32_t, U>) {
    ASSERT(value == static_cast<U>(0x9249249));
  } else {
    ASSERT(value == static_cast<U>(0x1249249249249249));
  }
}

template <std::unsigned_integral U, std::floating_point T>
HOSTDEV
TEST_CASE(mortonDecodeFloat)
{

  // NOLINTNEXTLINE justification: reduce clutter
  using namespace um2;

  T x;
  T y;
  mortonDecode<U, T>(static_cast<U>(0x0000000000000000), x, y);
  ASSERT_NEAR(x, static_cast<T>(0), static_cast<T>(1e-6));
  ASSERT_NEAR(y, static_cast<T>(0), static_cast<T>(1e-6));
  mortonDecode<U, T>(static_cast<U>(0xFFFFFFFFFFFFFFFF), x, y);
  ASSERT_NEAR(x, static_cast<T>(1), static_cast<T>(1e-6));
  ASSERT_NEAR(y, static_cast<T>(1), static_cast<T>(1e-6));
  mortonDecode<U, T>(static_cast<U>(0x5555555555555555), x, y);
  ASSERT_NEAR(x, static_cast<T>(1), static_cast<T>(1e-6));
  ASSERT_NEAR(y, static_cast<T>(0), static_cast<T>(1e-6));
  mortonDecode<U, T>(static_cast<U>(0xAAAAAAAAAAAAAAAA), x, y);
  ASSERT_NEAR(x, static_cast<T>(0), static_cast<T>(1e-6));
  ASSERT_NEAR(y, static_cast<T>(1), static_cast<T>(1e-6));

  // 3D
  T z;
  mortonDecode<U, T>(static_cast<U>(0x0000000000000000), x, y, z);
  ASSERT_NEAR(x, static_cast<T>(0), static_cast<T>(1e-6));
  ASSERT_NEAR(y, static_cast<T>(0), static_cast<T>(1e-6));
  ASSERT_NEAR(z, static_cast<T>(0), static_cast<T>(1e-6));
  if constexpr (std::same_as<uint32_t, U>) {
    mortonDecode<U, T>(static_cast<U>(0x3fffffff), x, y, z);
  } else {
    mortonDecode<U, T>(static_cast<U>(0x7fffffffffffffff), x, y, z);
  }
  ASSERT_NEAR(x, static_cast<T>(1), static_cast<T>(1e-6));
  ASSERT_NEAR(y, static_cast<T>(1), static_cast<T>(1e-6));
  ASSERT_NEAR(z, static_cast<T>(1), static_cast<T>(1e-6));
}

#if UM2_USE_CUDA
template <std::unsigned_integral U>
MAKE_CUDA_KERNEL(mortonEncode, U);

template <std::unsigned_integral U>
MAKE_CUDA_KERNEL(mortonDecode, U);

template <std::unsigned_integral U, std::floating_point T>
MAKE_CUDA_KERNEL(mortonEncodeFloat, U, T);

template <std::unsigned_integral U, std::floating_point T>
MAKE_CUDA_KERNEL(mortonDecodeFloat, U, T);
#endif

template <std::unsigned_integral U>
TEST_SUITE(morton)
{
  TEST_HOSTDEV(mortonEncode, 1, 1, U);
  TEST_HOSTDEV(mortonDecode, 1, 1, U);
}

template <std::unsigned_integral U, std::floating_point T>
TEST_SUITE(mortonFloat)
{
  TEST_HOSTDEV(mortonEncodeFloat, 1, 1, U, T);
  TEST_HOSTDEV(mortonDecodeFloat, 1, 1, U, T);
}

auto
main() -> int
{
  RUN_SUITE(morton<uint32_t>);
  RUN_SUITE(morton<uint64_t>);
  RUN_SUITE((mortonFloat<uint32_t, float>));
  RUN_SUITE((mortonFloat<uint32_t, double>));
  RUN_SUITE((mortonFloat<uint64_t, double>));
  return 0;
}
