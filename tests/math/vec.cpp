#include <um2/math/vec.hpp>

#include <um2/config.hpp>

#include "../test_macros.hpp"

// NOLINTNEXTLINE(misc-include-cleaner)
#include <concepts>

template <Int D, class T>
HOSTDEV constexpr auto
makeVec() -> um2::Vec<D, T>
{
  um2::Vec<D, T> v;
  for (Int i = 0; i < D; ++i) {
    v[i] = static_cast<T>(i + 1);
  }
  return v;
}

namespace
{
consteval auto
isPowerOf2(Int x) noexcept -> bool
{
  return (x & (x - 1)) == 0;
};
} // namespace

template <Int D, class T>
HOSTDEV
TEST_CASE(isSIMD)
{
  if constexpr (isPowerOf2(D)) {
    static_assert(alignof(um2::Vec<D, T>) == alignof(T) * D);
    static_assert(!um2::IsSIMDVec<T>::value);
    static_assert(um2::IsSIMDVec<um2::Vec<D, T>>::value);
    static_assert(um2::IsSIMDVec<um2::Vec<4, double>>::value);
    static_assert(!um2::IsSIMDVec<um2::Vec<3, double>>::value);
    static_assert(alignof(um2::Vec<4, double>) == alignof(double) * 4);
    static_assert(alignof(um2::Vec<3, double>) == alignof(double));
    static_assert(alignof(um2::Vec<4, um2::Vec<2, double>>) == alignof(double) * 8);
  }
}

template <Int D, class T>
HOSTDEV
TEST_CASE(accessor)
{
  um2::Vec<D, T> v = makeVec<D, T>();
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

template <Int D, class T>
HOSTDEV
TEST_CASE(compound_add)
{
  um2::Vec<D, T> v = makeVec<D, T>();
  um2::Vec<D, T> const v2 = makeVec<D, T>();
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

template <Int D, class T>
HOSTDEV
TEST_CASE(compound_sub)
{
  um2::Vec<D, T> v = makeVec<D, T>();
  um2::Vec<D, T> const v2 = makeVec<D, T>();
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

template <Int D, class T>
HOSTDEV
TEST_CASE(compound_mul)
{
  um2::Vec<D, T> v = makeVec<D, T>();
  um2::Vec<D, T> const v2 = makeVec<D, T>();
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

template <Int D, class T>
HOSTDEV
TEST_CASE(compound_div)
{
  um2::Vec<D, T> v = makeVec<D, T>();
  um2::Vec<D, T> const v2 = makeVec<D, T>();
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

template <Int D, class T>
HOSTDEV
TEST_CASE(scalar_assign)
{
  um2::Vec<D, T> v = makeVec<D, T>();
  v = 11;
  if constexpr (std::floating_point<T>) {
    for (Int i = 0; i < D; ++i) {
      ASSERT_NEAR(v[i], static_cast<T>(11), static_cast<T>(1e-6));
    }
  } else {
    for (Int i = 0; i < D; ++i) {
      ASSERT(v[i] == static_cast<T>(11));
    }
  }
}

template <Int D, class T>
HOSTDEV
TEST_CASE(compound_scalar_add)
{
  um2::Vec<D, T> v = makeVec<D, T>();
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

template <Int D, class T>
HOSTDEV
TEST_CASE(compound_scalar_sub)
{
  um2::Vec<D, T> v = makeVec<D, T>();
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

template <Int D, class T>
HOSTDEV
TEST_CASE(compound_scalar_mul)
{
  um2::Vec<D, T> v = makeVec<D, T>();
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

template <Int D, class T>
HOSTDEV
TEST_CASE(compound_scalar_div)
{
  um2::Vec<D, T> v = makeVec<D, T>();
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

template <Int D, class T>
HOSTDEV
TEST_CASE(zero)
{
  um2::Vec<D, T> const v = um2::Vec<D, T>::zero();
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

template <Int D, class T>
HOSTDEV
TEST_CASE(min)
{
  um2::Vec<D, T> const v0 = makeVec<D, T>();
  um2::Vec<D, T> v1 = makeVec<D, T>();
  for (Int i = 0; i < D; ++i) {
    v1[i] += 1;
  }
  v1 = um2::min(v0, v1);
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

template <Int D, class T>
HOSTDEV
TEST_CASE(max)
{
  um2::Vec<D, T> v0 = makeVec<D, T>();
  um2::Vec<D, T> v1 = makeVec<D, T>();
  for (Int i = 0; i < D; ++i) {
    v0[i] += 1;
  }
  v1 = um2::max(v0, v1);
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

template <Int D, class T>
HOSTDEV
TEST_CASE(dot)
{
  um2::Vec<D, T> const v = makeVec<D, T>();
  T dot = um2::dot(v, v);
  if constexpr (std::floating_point<T>) {
    ASSERT_NEAR(dot, static_cast<T>(D * (D + 1) * (2 * D + 1)) / 6, static_cast<T>(1e-6));
  } else {
    ASSERT(dot == static_cast<T>(D * (D + 1) * (2 * D + 1) / 6));
  }
}

template <Int D, class T>
HOSTDEV
TEST_CASE(squaredNorm)
{
  um2::Vec<D, T> const v = makeVec<D, T>();
  T norm2 = um2::squaredNorm(v);
  if constexpr (std::floating_point<T>) {
    ASSERT_NEAR(norm2, static_cast<T>(D * (D + 1) * (2 * D + 1)) / 6,
                static_cast<T>(1e-6));
  } else {
    ASSERT(norm2 == static_cast<T>(D * (D + 1) * (2 * D + 1) / 6));
  }
}

template <Int D, class T>
HOSTDEV
TEST_CASE(norm)
{
  um2::Vec<D, T> const v = makeVec<D, T>();
  T norm = um2::norm(v);
  T ref = um2::sqrt(static_cast<T>(D * (D + 1) * (2 * D + 1)) / 6);
  ASSERT_NEAR(norm, ref, static_cast<T>(1e-6));
}

template <Int D, class T>
HOSTDEV
TEST_CASE(normalized)
{
  um2::Vec<D, T> const v = makeVec<D, T>();
  um2::Vec<D, T> const vn = um2::normalized(v);
  ASSERT_NEAR(um2::norm(vn), 1, static_cast<T>(1e-6));
}

template <Int D, class T>
HOSTDEV
TEST_CASE(cross)
{
  if constexpr (D == 3) {
    um2::Vec<D, T> const v0(1, 2, 3);
    um2::Vec<D, T> const v1(2, 3, 4);
    um2::Vec<D, T> v = um2::cross(v0, v1);
    ASSERT_NEAR(v[0], -1, static_cast<T>(1e-6));
    ASSERT_NEAR(v[1], 2, static_cast<T>(1e-6));
    ASSERT_NEAR(v[2], -1, static_cast<T>(1e-6));
  } else if constexpr (D == 2) {
    um2::Vec2<T> const v0(1, 2);
    um2::Vec2<T> const v1(3, 4);
    T x = um2::cross(v0, v1);
    ASSERT_NEAR(x, -2, static_cast<T>(1e-6));
  }
}

template <Int D, class T>
HOSTDEV
TEST_CASE(isApprox)
{
  um2::Vec<D, T> const v1 = um2::Vec<D, T>::zero();
  um2::Vec<D, T> v2 = um2::Vec<D, T>::zero() + 1;
  // Trivial equality
  ASSERT(v1.isApprox(v1));
  // Trivial inequality
  ASSERT(!v1.isApprox(v2));
  // Non-trivial equality
  v2 = v1;
  v2[0] += um2::epsDistance<T>() / 2;
  ASSERT(v1.isApprox(v2));
  // Non-trivial inequality
  v2[0] += um2::epsDistance<T>();
  ASSERT(!v1.isApprox(v2));
}

template <Int D, class T>
HOSTDEV
TEST_CASE(relational)
{
  um2::Vec<D, T> const v = makeVec<D, T>();
  um2::Vec<D, T> v2 = makeVec<D, T>();
  ASSERT(v == v2);
  ASSERT(v <= v2);
  ASSERT(v >= v2);
  v2[1] += 1;
  ASSERT(v != v2);
  ASSERT(v < v2);
  ASSERT(v <= v2);
  ASSERT(v2 > v);
  ASSERT(v2 >= v);
}

template <Int D, class T>
HOSTDEV
TEST_CASE(add)
{
  um2::Vec<D, T> v = makeVec<D, T>();
  um2::Vec<D, T> const v2 = makeVec<D, T>();
  v = v + v2;
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

template <Int D, class T>
HOSTDEV
TEST_CASE(sub)
{
  um2::Vec<D, T> v = makeVec<D, T>();
  um2::Vec<D, T> const v2 = makeVec<D, T>();
  v = v - v2;
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

template <Int D, class T>
HOSTDEV
TEST_CASE(mul)
{
  um2::Vec<D, T> v = makeVec<D, T>();
  um2::Vec<D, T> const v2 = makeVec<D, T>();
  v = v * v2;
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

template <Int D, class T>
HOSTDEV
TEST_CASE(div)
{
  um2::Vec<D, T> v = makeVec<D, T>();
  um2::Vec<D, T> const v2 = makeVec<D, T>();
  v = v / v2;
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

template <Int D, class T>
HOSTDEV
TEST_CASE(scalar_add)
{
  um2::Vec<D, T> const v = makeVec<D, T>();
  auto const vl = 2 + v;
  auto const vr = v + 2;
  if constexpr (std::floating_point<T>) {
    for (Int i = 0; i < D; ++i) {
      ASSERT_NEAR(vl[i], static_cast<T>(i + 3), static_cast<T>(1e-6));
      ASSERT_NEAR(vr[i], static_cast<T>(i + 3), static_cast<T>(1e-6));
    }
  } else {
    for (Int i = 0; i < D; ++i) {
      ASSERT(vl[i] == static_cast<T>(i + 3));
      ASSERT(vr[i] == static_cast<T>(i + 3));
    }
  }
}

template <Int D, class T>
HOSTDEV
TEST_CASE(scalar_sub)
{
  um2::Vec<D, T> const v = makeVec<D, T>();
  auto const vl = v - 2;
  if constexpr (std::floating_point<T>) {
    for (Int i = 0; i < D; ++i) {
      ASSERT_NEAR(vl[i], static_cast<T>(i - 1), static_cast<T>(1e-6));
    }
  } else {
    for (Int i = 0; i < D; ++i) {
      ASSERT(vl[i] == static_cast<T>(i - 1));
    }
  }
}

template <Int D, class T>
HOSTDEV
TEST_CASE(scalar_mul)
{
  um2::Vec<D, T> const v = makeVec<D, T>();
  auto const vl = 2 * v;
  auto const vr = v * 2;
  if constexpr (std::floating_point<T>) {
    for (Int i = 0; i < D; ++i) {
      ASSERT_NEAR(vl[i], static_cast<T>(2 * (i + 1)), static_cast<T>(1e-6));
      ASSERT_NEAR(vr[i], static_cast<T>(2 * (i + 1)), static_cast<T>(1e-6));
    }
  } else {
    for (Int i = 0; i < D; ++i) {
      ASSERT(vl[i] == static_cast<T>(2 * (i + 1)));
      ASSERT(vr[i] == static_cast<T>(2 * (i + 1)));
    }
  }
}

template <Int D, class T>
HOSTDEV
TEST_CASE(scalar_div)
{
  um2::Vec<D, T> const v = makeVec<D, T>();
  auto const vl = v / 2;
  if constexpr (std::floating_point<T>) {
    for (Int i = 0; i < D; ++i) {
      ASSERT_NEAR(vl[i], static_cast<T>(i + 1) / 2, static_cast<T>(1e-6));
    }
  } else {
    for (Int i = 0; i < D; ++i) {
      ASSERT(vl[i] == static_cast<T>(i + 1) / 2);
    }
  }
}

//=============================================================================
// CUDA
//=============================================================================

#if UM2_USE_CUDA
template <Int D, class T>
MAKE_CUDA_KERNEL(isSIMD, D, T);

template <Int D, class T>
MAKE_CUDA_KERNEL(accessor, D, T);

template <Int D, class T>
MAKE_CUDA_KERNEL(compound_add, D, T);

template <Int D, class T>
MAKE_CUDA_KERNEL(compound_sub, D, T);

template <Int D, class T>
MAKE_CUDA_KERNEL(compound_mul, D, T);

template <Int D, class T>
MAKE_CUDA_KERNEL(compound_div, D, T);

template <Int D, class T>
MAKE_CUDA_KERNEL(scalar_assign, D, T);

template <Int D, class T>
MAKE_CUDA_KERNEL(compound_scalar_add, D, T);

template <Int D, class T>
MAKE_CUDA_KERNEL(compound_scalar_sub, D, T);

template <Int D, class T>
MAKE_CUDA_KERNEL(compound_scalar_mul, D, T);

template <Int D, class T>
MAKE_CUDA_KERNEL(compound_scalar_div, D, T);

template <Int D, class T>
MAKE_CUDA_KERNEL(zero, D, T);

template <Int D, class T>
MAKE_CUDA_KERNEL(min, D, T);

template <Int D, class T>
MAKE_CUDA_KERNEL(max, D, T);

template <Int D, class T>
MAKE_CUDA_KERNEL(dot, D, T);

template <Int D, class T>
MAKE_CUDA_KERNEL(squaredNorm, D, T);

template <Int D, class T>
MAKE_CUDA_KERNEL(norm, D, T);

template <Int D, class T>
MAKE_CUDA_KERNEL(normalized, D, T);

template <Int D, class T>
MAKE_CUDA_KERNEL(cross, D, T);

template <Int D, class T>
MAKE_CUDA_KERNEL(isApprox, D, T);

template <Int D, class T>
MAKE_CUDA_KERNEL(relational, D, T);

template <Int D, class T>
MAKE_CUDA_KERNEL(add, D, T);

template <Int D, class T>
MAKE_CUDA_KERNEL(sub, D, T);

template <Int D, class T>
MAKE_CUDA_KERNEL(mul, D, T);

template <Int D, class T>
MAKE_CUDA_KERNEL(div, D, T);

template <Int D, class T>
MAKE_CUDA_KERNEL(scalar_add, D, T);

template <Int D, class T>
MAKE_CUDA_KERNEL(scalar_sub, D, T);

template <Int D, class T>
MAKE_CUDA_KERNEL(scalar_mul, D, T);

template <Int D, class T>
MAKE_CUDA_KERNEL(scalar_div, D, T);
#endif

template <Int D, class T>
TEST_SUITE(vec)
{
  TEST_HOSTDEV(isSIMD, D, T);
  TEST_HOSTDEV(accessor, D, T);
  TEST_HOSTDEV(compound_add, D, T);
  TEST_HOSTDEV(compound_sub, D, T);
  TEST_HOSTDEV(compound_mul, D, T);
  TEST_HOSTDEV(compound_div, D, T);
  TEST_HOSTDEV(scalar_assign, D, T);
  TEST_HOSTDEV(compound_scalar_add, D, T);
  TEST_HOSTDEV(compound_scalar_sub, D, T);
  TEST_HOSTDEV(compound_scalar_mul, D, T);
  TEST_HOSTDEV(compound_scalar_div, D, T);
  TEST_HOSTDEV(zero, D, T);
  TEST_HOSTDEV(min, D, T);
  TEST_HOSTDEV(max, D, T);
  TEST_HOSTDEV(dot, D, T);
  TEST_HOSTDEV(squaredNorm, D, T);
  if constexpr (std::floating_point<T>) {
    TEST_HOSTDEV(norm, D, T);
    TEST_HOSTDEV(normalized, D, T);
    TEST_HOSTDEV(cross, D, T);
    TEST_HOSTDEV(isApprox, D, T);
  }
  if constexpr (std::integral<T>) {
    TEST_HOSTDEV(relational, D, T);
  }
  TEST_HOSTDEV(add, D, T);
  TEST_HOSTDEV(sub, D, T);
  TEST_HOSTDEV(mul, D, T);
  TEST_HOSTDEV(div, D, T);
  TEST_HOSTDEV(scalar_add, D, T);
  TEST_HOSTDEV(scalar_sub, D, T);
  TEST_HOSTDEV(scalar_mul, D, T);
  TEST_HOSTDEV(scalar_div, D, T);
}

auto
main() -> int
{
  RUN_SUITE((vec<2, float>));
  RUN_SUITE((vec<2, double>));
  RUN_SUITE((vec<2, Int>));

  RUN_SUITE((vec<3, float>));
  RUN_SUITE((vec<3, double>));
  RUN_SUITE((vec<3, Int>));

  RUN_SUITE((vec<4, float>));
  RUN_SUITE((vec<4, double>));
  RUN_SUITE((vec<4, Int>));

  RUN_SUITE((vec<8, float>));
  RUN_SUITE((vec<8, double>));
  RUN_SUITE((vec<8, Int>));

  RUN_SUITE((vec<16, float>));
  RUN_SUITE((vec<16, double>));
  RUN_SUITE((vec<16, Int>));

  RUN_SUITE((vec<32, float>));
  RUN_SUITE((vec<32, double>));
  RUN_SUITE((vec<32, Int>));

  return 0;
}
