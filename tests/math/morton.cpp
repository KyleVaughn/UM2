#include "../test_framework.hpp"
#include <um2/math/morton.hpp>

template <typename U>
UM2_HOSTDEV
TEST_CASE(mortonEncode)
{
  EXPECT_EQ(um2::mortonEncode(static_cast<U>(0), static_cast<U>(0)), 0);
  EXPECT_EQ(um2::mortonEncode(static_cast<U>(1), static_cast<U>(0)), 1);
  EXPECT_EQ(um2::mortonEncode(static_cast<U>(0), static_cast<U>(1)), 2);
  EXPECT_EQ(um2::mortonEncode(static_cast<U>(1), static_cast<U>(1)), 3);
  EXPECT_EQ(um2::mortonEncode(static_cast<U>(2), static_cast<U>(2)), 12);
  EXPECT_EQ(um2::mortonEncode(static_cast<U>(3), static_cast<U>(3)), 15);

  EXPECT_EQ(um2::mortonEncode(static_cast<U>(0), static_cast<U>(0), static_cast<U>(0)),
            0);
  EXPECT_EQ(um2::mortonEncode(static_cast<U>(1), static_cast<U>(0), static_cast<U>(0)),
            1);
  EXPECT_EQ(um2::mortonEncode(static_cast<U>(0), static_cast<U>(1), static_cast<U>(0)),
            2);
  EXPECT_EQ(um2::mortonEncode(static_cast<U>(1), static_cast<U>(1), static_cast<U>(0)),
            3);
  EXPECT_EQ(um2::mortonEncode(static_cast<U>(0), static_cast<U>(0), static_cast<U>(1)),
            4);
  EXPECT_EQ(um2::mortonEncode(static_cast<U>(1), static_cast<U>(0), static_cast<U>(1)),
            5);
  EXPECT_EQ(um2::mortonEncode(static_cast<U>(0), static_cast<U>(1), static_cast<U>(1)),
            6);
  EXPECT_EQ(um2::mortonEncode(static_cast<U>(1), static_cast<U>(1), static_cast<U>(1)),
            7);
  EXPECT_EQ(um2::mortonEncode(static_cast<U>(2), static_cast<U>(0), static_cast<U>(0)),
            8);
  EXPECT_EQ(um2::mortonEncode(static_cast<U>(2), static_cast<U>(2), static_cast<U>(2)),
            56);
  EXPECT_EQ(um2::mortonEncode(static_cast<U>(3), static_cast<U>(3), static_cast<U>(3)),
            63);
}

template <typename U>
UM2_HOSTDEV
TEST_CASE(mortonDecode)
{
  U x;
  U y;
  um2::mortonDecode(static_cast<U>(0), x, y);
  EXPECT_TRUE(x == 0 && y == 0);
  um2::mortonDecode(static_cast<U>(1), x, y);
  EXPECT_TRUE(x == 1 && y == 0);
  um2::mortonDecode(static_cast<U>(2), x, y);
  EXPECT_TRUE(x == 0 && y == 1);
  um2::mortonDecode(static_cast<U>(3), x, y);
  EXPECT_TRUE(x == 1 && y == 1);
  um2::mortonDecode(static_cast<U>(12), x, y);
  EXPECT_TRUE(x == 2 && y == 2);
  um2::mortonDecode(static_cast<U>(15), x, y);
  EXPECT_TRUE(x == 3 && y == 3);

  U z;
  um2::mortonDecode(static_cast<U>(0), x, y, z);
  EXPECT_TRUE(x == 0 && y == 0 && z == 0);
  um2::mortonDecode(static_cast<U>(1), x, y, z);
  EXPECT_TRUE(x == 1 && y == 0 && z == 0);
  um2::mortonDecode(static_cast<U>(2), x, y, z);
  EXPECT_TRUE(x == 0 && y == 1 && z == 0);
  um2::mortonDecode(static_cast<U>(3), x, y, z);
  EXPECT_TRUE(x == 1 && y == 1 && z == 0);
  um2::mortonDecode(static_cast<U>(4), x, y, z);
  EXPECT_TRUE(x == 0 && y == 0 && z == 1);
  um2::mortonDecode(static_cast<U>(5), x, y, z);
  EXPECT_TRUE(x == 1 && y == 0 && z == 1);
  um2::mortonDecode(static_cast<U>(6), x, y, z);
  EXPECT_TRUE(x == 0 && y == 1 && z == 1);
  um2::mortonDecode(static_cast<U>(7), x, y, z);
  EXPECT_TRUE(x == 1 && y == 1 && z == 1);
  um2::mortonDecode(static_cast<U>(56), x, y, z);
  EXPECT_TRUE(x == 2 && y == 2 && z == 2);
}

template <typename U, typename T>
UM2_HOSTDEV
TEST_CASE(normalizedMortonEncode)
{
  T xscale = static_cast<T>(2);
  T yscale = static_cast<T>(4);
  T xscale_inv = static_cast<T>(1) / xscale;
  T yscale_inv = static_cast<T>(1) / yscale;
  U value = um2::normalizedMortonEncode<U, T>(static_cast<T>(0), static_cast<T>(0),
                                              xscale_inv, yscale_inv);
  EXPECT_EQ(value, static_cast<U>(0x0000000000000000));
  value = um2::normalizedMortonEncode<U, T>(static_cast<T>(2), static_cast<T>(4),
                                            xscale_inv, yscale_inv);
  EXPECT_EQ(value, static_cast<U>(0xffffffffffffffff));
  value = um2::normalizedMortonEncode<U, T>(static_cast<T>(2), static_cast<T>(0),
                                            xscale_inv, yscale_inv);
  EXPECT_EQ(value, static_cast<U>(0x5555555555555555));
  value = um2::normalizedMortonEncode<U, T>(static_cast<T>(0), static_cast<T>(4),
                                            xscale_inv, yscale_inv);
  EXPECT_EQ(value, static_cast<U>(0xaaaaaaaaaaaaaaaa));

  T zscale = static_cast<T>(8);
  T zscale_inv = static_cast<T>(1) / zscale;
  value = um2::normalizedMortonEncode<U, T>(static_cast<T>(0), static_cast<T>(0),
                                            static_cast<T>(0), xscale_inv, yscale_inv,
                                            zscale_inv);
  EXPECT_EQ(value, static_cast<U>(0x0000000000000000));
  value = um2::normalizedMortonEncode<U, T>(static_cast<T>(2), static_cast<T>(4),
                                            static_cast<T>(8), xscale_inv, yscale_inv,
                                            zscale_inv);
  if constexpr (std::same_as<uint32_t, U>) {
    EXPECT_EQ(value, static_cast<U>(0x3fffffff));
  } else {
    EXPECT_EQ(value, static_cast<U>(0x7fffffffffffffff));
  }
  value = um2::normalizedMortonEncode<U, T>(static_cast<T>(2), static_cast<T>(0),
                                            static_cast<T>(0), xscale_inv, yscale_inv,
                                            zscale_inv);
  if constexpr (std::same_as<uint32_t, U>) {
    EXPECT_EQ(value, static_cast<U>(0x9249249));
  } else {
    EXPECT_EQ(value, static_cast<U>(0x1249249249249249));
  }
}

template <typename U, typename T>
UM2_HOSTDEV
TEST_CASE(normalizedMortonDecode)
{
  T xscale = static_cast<T>(2);
  T yscale = static_cast<T>(4);
  T x;
  T y;
  um2::normalizedMortonDecode<U, T>(static_cast<U>(0x0000000000000000), x, y, xscale,
                                    yscale);
  EXPECT_NEAR(x, static_cast<T>(0), 1e-6);
  EXPECT_NEAR(y, static_cast<T>(0), 1e-6);
  um2::normalizedMortonDecode<U, T>(static_cast<U>(0xffffffffffffffff), x, y, xscale,
                                    yscale);
  EXPECT_NEAR(x, static_cast<T>(2), 1e-6);
  EXPECT_NEAR(y, static_cast<T>(4), 1e-6);
  um2::normalizedMortonDecode<U, T>(static_cast<U>(0x5555555555555555), x, y, xscale,
                                    yscale);
  EXPECT_NEAR(x, static_cast<T>(2), 1e-6);
  EXPECT_NEAR(y, static_cast<T>(0), 1e-6);
  um2::normalizedMortonDecode<U, T>(static_cast<U>(0xaaaaaaaaaaaaaaaa), x, y, xscale,
                                    yscale);
  EXPECT_NEAR(x, static_cast<T>(0), 1e-6);
  EXPECT_NEAR(y, static_cast<T>(4), 1e-6);

  T zscale = static_cast<T>(8);
  T z;
  um2::normalizedMortonDecode<U, T>(static_cast<U>(0x0000000000000000), x, y, z, xscale,
                                    yscale, zscale);
  EXPECT_NEAR(x, static_cast<T>(0), 1e-6);
  EXPECT_NEAR(y, static_cast<T>(0), 1e-6);
  EXPECT_NEAR(z, static_cast<T>(0), 1e-6);
  if constexpr (std::same_as<uint32_t, U>) {
    um2::normalizedMortonDecode<U, T>(static_cast<U>(0x3fffffff), x, y, z, xscale, yscale,
                                      zscale);
    EXPECT_NEAR(x, static_cast<T>(2), 1e-6);
    EXPECT_NEAR(y, static_cast<T>(4), 1e-6);
    EXPECT_NEAR(z, static_cast<T>(8), 1e-6);
  } else {
    um2::normalizedMortonDecode<U, T>(static_cast<U>(0x7fffffffffffffff), x, y, z, xscale,
                                      yscale, zscale);
    EXPECT_NEAR(x, static_cast<T>(2), 1e-6);
    EXPECT_NEAR(y, static_cast<T>(4), 1e-6);
    EXPECT_NEAR(z, static_cast<T>(8), 1e-6);
  }
}

#if UM2_ENABLE_CUDA
template <typename U>
MAKE_CUDA_KERNEL(mortonEncode, U);

template <typename U>
MAKE_CUDA_KERNEL(mortonDecode, U);

template <typename U, typename T>
MAKE_CUDA_KERNEL(normalizedMortonEncode, U, T);

template <typename U, typename T>
MAKE_CUDA_KERNEL(normalizedMortonDecode, U, T);
#endif

template <typename U>
TEST_SUITE(morton)
{
  TEST_HOSTDEV(mortonEncode, 1, 1, U);
  TEST_HOSTDEV(mortonDecode, 1, 1, U);
  if constexpr (std::same_as<U, uint32_t>) {
    TEST_HOSTDEV(normalizedMortonEncode, 1, 1, U, float);
    TEST_HOSTDEV(normalizedMortonDecode, 1, 1, U, float);
  } else {
    TEST_HOSTDEV(normalizedMortonEncode, 1, 1, U, double);
    TEST_HOSTDEV(normalizedMortonDecode, 1, 1, U, double);
  }
}

auto
main() -> int
{
  RUN_TESTS(morton<uint32_t>);
  RUN_TESTS(morton<uint64_t>);
  return 0;
}
