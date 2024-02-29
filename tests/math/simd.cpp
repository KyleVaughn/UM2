#include <um2/math/simd.hpp>

#include "../test_macros.hpp"

// We know the ABI will be different based upon if the CPU supports
// SSE, AVX, etc. or the compiler options.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpsabi"

template <Int D, typename T>
HOSTDEV constexpr auto
makeSIMD() -> um2::SIMD<D, T>
{
  um2::SIMD<D, T> v;
  for (Int i = 0; i < D; ++i) {
    v[i] = static_cast<T>(i + 1);
  }
  return v;
}

template <Int D, typename T>
HOSTDEV
TEST_CASE(accessor)
{
  um2::SIMD<D, T> v; //= makeSIMD<D, T>();
  for (Int i = 0; i < D; ++i) {
    v[i] = static_cast<T>(i + 1);
  }
  if constexpr (std::floating_point<T>) {
    for (Int i = 0; i < D; ++i) {
      ASSERT_NEAR(v[i], static_cast<T>(i + 1), static_cast<T>(1e-6));
    }
  } else {
    for (Int i = 0; i < D; ++i) {
      ASSERT(v[i] == static_cast<T>(i + 1));
    }
  }
}

template <Int D, typename T>
HOSTDEV
TEST_CASE(compound_add)
{
  um2::SIMD<D, T> v = makeSIMD<D, T>();
  um2::SIMD<D, T> const v2 = makeSIMD<D, T>();
  v += v2;
  if constexpr (std::floating_point<T>) {
    for (Int i = 0; i < D; ++i) {
      ASSERT_NEAR(v[i], static_cast<T>(2 * (i + 1)), static_cast<T>(1e-6));
    }
  } else {
    for (Int i = 0; i < D; ++i) {
      ASSERT(v[i] == static_cast<T>(2 * (i + 1)));
    }
  }
}

template <Int D, typename T>
HOSTDEV
TEST_CASE(compound_sub)
{
  um2::SIMD<D, T> v = makeSIMD<D, T>();
  um2::SIMD<D, T> const v2 = makeSIMD<D, T>();
  v -= v2;
  if constexpr (std::floating_point<T>) {
    for (Int i = 0; i < D; ++i) {
      ASSERT_NEAR(v[i], 0, static_cast<T>(1e-6));
    }
  } else {
    for (Int i = 0; i < D; ++i) {
      ASSERT(v[i] == 0);
    }
  }
}

template <Int D, typename T>
HOSTDEV
TEST_CASE(compound_mul)
{
  um2::SIMD<D, T> v = makeSIMD<D, T>();
  um2::SIMD<D, T> const v2 = makeSIMD<D, T>();
  v *= v2;
  if constexpr (std::floating_point<T>) {
    for (Int i = 0; i < D; ++i) {
      ASSERT_NEAR(v[i], static_cast<T>((i + 1) * (i + 1)), static_cast<T>(1e-6));
    }
  } else {
    for (Int i = 0; i < D; ++i) {
      ASSERT(v[i] == static_cast<T>((i + 1) * (i + 1)));
    }
  }
}

template <Int D, typename T>
HOSTDEV
TEST_CASE(compound_div)
{
  um2::SIMD<D, T> v = makeSIMD<D, T>();
  um2::SIMD<D, T> const v2 = makeSIMD<D, T>();
  v /= v2;
  if constexpr (std::floating_point<T>) {
    for (Int i = 0; i < D; ++i) {
      ASSERT_NEAR(v[i], 1, static_cast<T>(1e-6));
    }
  } else {
    for (Int i = 0; i < D; ++i) {
      ASSERT(v[i] == 1);
    }
  }
}

template <Int D, typename T>
HOSTDEV
TEST_CASE(compound_scalar_add)
{
  um2::SIMD<D, T> v = makeSIMD<D, T>();
  v += 2;
  if constexpr (std::floating_point<T>) {
    for (Int i = 0; i < D; ++i) {
      ASSERT_NEAR(v[i], static_cast<T>(i + 3), static_cast<T>(1e-6));
    }
  } else {
    for (Int i = 0; i < D; ++i) {
      ASSERT(v[i] == static_cast<T>(i + 3));
    }
  }
}

template <Int D, typename T>
HOSTDEV
TEST_CASE(compound_scalar_sub)
{
  um2::SIMD<D, T> v = makeSIMD<D, T>();
  v -= 2;
  if constexpr (std::floating_point<T>) {
    for (Int i = 0; i < D; ++i) {
      ASSERT_NEAR(v[i], static_cast<T>(i - 1), static_cast<T>(1e-6));
    }
  } else {
    for (Int i = 0; i < D; ++i) {
      ASSERT(v[i] == static_cast<T>(i - 1));
    }
  }
}

template <Int D, typename T>
HOSTDEV
TEST_CASE(compound_scalar_mul)
{
  um2::SIMD<D, T> v = makeSIMD<D, T>();
  v *= 2;
  if constexpr (std::floating_point<T>) {
    for (Int i = 0; i < D; ++i) {
      ASSERT_NEAR(v[i], static_cast<T>(2 * (i + 1)), static_cast<T>(1e-6));
    }
  } else {
    for (Int i = 0; i < D; ++i) {
      ASSERT(v[i] == static_cast<T>(2 * (i + 1)));
    }
  }
}

template <Int D, typename T>
HOSTDEV
TEST_CASE(compound_scalar_div)
{
  um2::SIMD<D, T> v = makeSIMD<D, T>();
  v /= 2;
  if constexpr (std::floating_point<T>) {
    for (Int i = 0; i < D; ++i) {
      ASSERT_NEAR(v[i], static_cast<T>(i + 1) / 2, static_cast<T>(1e-6));
    }
  } else {
    for (Int i = 0; i < D; ++i) {
      ASSERT(v[i] == static_cast<T>(i + 1) / 2);
    }
  }
}

template <Int D, typename T>
HOSTDEV
TEST_CASE(min)
{
  um2::SIMD<D, T> const v0 = makeSIMD<D, T>();
  um2::SIMD<D, T> v1 = makeSIMD<D, T>();
  v1 += 1;
  v1 = um2::min<D>(v1, v0);
  if constexpr (std::floating_point<T>) {
    for (Int i = 0; i < D; ++i) {
      ASSERT_NEAR(v1[i], static_cast<T>(i + 1), static_cast<T>(1e-6));
    }
  } else {
    for (Int i = 0; i < D; ++i) {
      ASSERT(v1[i] == static_cast<T>(i + 1));
    }
  }
}

template <Int D, typename T>
HOSTDEV
TEST_CASE(max)
{
  um2::SIMD<D, T> const v0 = makeSIMD<D, T>();
  um2::SIMD<D, T> v1 = makeSIMD<D, T>();
  v1 += 1;
  v1 = um2::max<D>(v0, v1);
  if constexpr (std::floating_point<T>) {
    for (Int i = 0; i < D; ++i) {
      ASSERT_NEAR(v1[i], static_cast<T>(i + 2), static_cast<T>(1e-6));
    }
  } else {
    for (Int i = 0; i < D; ++i) {
      ASSERT(v1[i] == static_cast<T>(i + 2));
    }
  }
}

template <Int D, typename T>
HOSTDEV
TEST_CASE(dot)
{
  um2::SIMD<D, T> const v = makeSIMD<D, T>();
  // Compiler has trouble with infering D in the template argument
  T dot = um2::dot<D>(v, v);
  if constexpr (std::floating_point<T>) {
    ASSERT_NEAR(dot, static_cast<T>(D * (D + 1) * (2 * D + 1)) / 6, static_cast<T>(1e-6));
  } else {
    ASSERT(dot == static_cast<T>(D * (D + 1) * (2 * D + 1) / 6));
  }
}

template <Int D, typename T>
HOSTDEV
TEST_CASE(squaredNorm)
{
  um2::SIMD<D, T> const v = makeSIMD<D, T>();
  T norm2 = um2::squaredNorm<D>(v);
  if constexpr (std::floating_point<T>) {
    ASSERT_NEAR(norm2, static_cast<T>(D * (D + 1) * (2 * D + 1)) / 6,
                static_cast<T>(1e-6));
  } else {
    ASSERT(norm2 == static_cast<T>(D * (D + 1) * (2 * D + 1) / 6));
  }
}

template <Int D, typename T>
HOSTDEV
TEST_CASE(norm)
{
  um2::SIMD<D, T> const v = makeSIMD<D, T>();
  T norm = um2::norm<D>(v);
  T ref = um2::sqrt(static_cast<T>(D * (D + 1) * (2 * D + 1)) / 6);
  ASSERT_NEAR(norm, ref, static_cast<T>(1e-6));
}

template <Int D, typename T>
HOSTDEV
TEST_CASE(normalized)
{
  um2::SIMD<D, T> const v = makeSIMD<D, T>();
  um2::SIMD<D, T> const vn = um2::normalized<D>(v);
  ASSERT_NEAR(um2::norm<D>(vn), 1, static_cast<T>(1e-6));
}

template <Int D, typename T>
HOSTDEV
TEST_CASE(cross)
{
  if constexpr (D == 2) {
    um2::SIMD<2, T> v0;
    v0[0] = 1;
    v0[1] = 2;
    um2::SIMD<2, T> v1;
    v1[0] = 3;
    v1[1] = 4;
    T x = um2::cross(v0, v1);
    ASSERT_NEAR(x, -2, static_cast<T>(1e-6));
  }
}

//=============================================================================
// CUDA
//=============================================================================

#if UM2_USE_CUDA
template <Int D, typename T>
MAKE_CUDA_KERNEL(accessor, D, T);

template <Int D, typename T>
MAKE_CUDA_KERNEL(unary_minus, D, T);

template <Int D, typename T>
MAKE_CUDA_KERNEL(compound_add, D, T);

template <Int D, typename T>
MAKE_CUDA_KERNEL(compound_sub, D, T);

template <Int D, typename T>
MAKE_CUDA_KERNEL(compound_mul, D, T);

template <Int D, typename T>
MAKE_CUDA_KERNEL(compound_div, D, T);

template <Int D, typename T>
MAKE_CUDA_KERNEL(compound_scalar_add, D, T);

template <Int D, typename T>
MAKE_CUDA_KERNEL(compound_scalar_sub, D, T);

template <Int D, typename T>
MAKE_CUDA_KERNEL(compound_scalar_mul, D, T);

template <Int D, typename T>
MAKE_CUDA_KERNEL(compound_scalar_div, D, T);

template <Int D, typename T>
MAKE_CUDA_KERNEL(min, D, T);

template <Int D, typename T>
MAKE_CUDA_KERNEL(max, D, T);

template <Int D, typename T>
MAKE_CUDA_KERNEL(dot, D, T);

template <Int D, typename T>
MAKE_CUDA_KERNEL(cross, D, T);

template <Int D, typename T>
MAKE_CUDA_KERNEL(squaredNorm, D, T);

template <Int D, typename T>
MAKE_CUDA_KERNEL(norm, D, T);

template <Int D, typename T>
MAKE_CUDA_KERNEL(normalized, D, T);

#endif

template <Int D, typename T>
TEST_SUITE(simd)
{
  TEST_HOSTDEV(accessor, D, T);
  TEST_HOSTDEV(compound_add, D, T);
  TEST_HOSTDEV(compound_sub, D, T);
  TEST_HOSTDEV(compound_mul, D, T);
  TEST_HOSTDEV(compound_div, D, T);
  TEST_HOSTDEV(compound_scalar_add, D, T);
  TEST_HOSTDEV(compound_scalar_sub, D, T);
  TEST_HOSTDEV(compound_scalar_mul, D, T);
  TEST_HOSTDEV(compound_scalar_div, D, T);
  TEST_HOSTDEV(min, D, T);
  TEST_HOSTDEV(max, D, T);
  TEST_HOSTDEV(dot, D, T);
  TEST_HOSTDEV(squaredNorm, D, T);
  if constexpr (std::floating_point<T>) {
    TEST_HOSTDEV(norm, D, T);
    TEST_HOSTDEV(normalized, D, T);
    TEST_HOSTDEV(cross, D, T);
  }
}

auto
main() -> int
{
  RUN_SUITE((simd<2, Float>));
  RUN_SUITE((simd<2, Int>));

  RUN_SUITE((simd<4, Float>));
  RUN_SUITE((simd<4, Int>));

  RUN_SUITE((simd<8, Float>));
  RUN_SUITE((simd<8, Int>));

  RUN_SUITE((simd<16, Float>));
  RUN_SUITE((simd<16, Int>));

  RUN_SUITE((simd<32, Float>));
  RUN_SUITE((simd<32, Int>));

  return 0;
}

#pragma clang diagnostic pop
