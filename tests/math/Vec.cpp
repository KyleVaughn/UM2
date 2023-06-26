#include <um2/math/Vec.hpp>

#include "../test_macros.hpp"

template <Size D, typename T>
HOSTDEV static constexpr auto
makeVec() -> um2::Vec<D, T>
{
  um2::Vec<D, T> v;
  for (Size i = 0; i < D; ++i) {
    v[i] = static_cast<T>(i + 1);
  }
  return v;
}

template <Size D, typename T>
HOSTDEV
TEST_CASE(accessor)
{
  um2::Vec<D, T> v = makeVec<D, T>();
  if constexpr (std::floating_point<T>) {
    for (Size i = 0; i < D; ++i) {
      ASSERT_NEAR(v[i], static_cast<T>(i + 1), static_cast<T>(1e-6));
    }
  } else {
    for (Size i = 0; i < D; ++i) {
      assert(v[i] == static_cast<T>(i + 1));
    }
  }
}

template <Size D, typename T>
HOSTDEV
TEST_CASE(unary_minus)
{
  um2::Vec<D, T> v = makeVec<D, T>();
  um2::Vec<D, T> v2 = -v;
  if constexpr (std::floating_point<T>) {
    for (Size i = 0; i < D; ++i) {
      ASSERT_NEAR(v2[i], -static_cast<T>(i + 1), static_cast<T>(1e-6));
    }
  } else {
    for (Size i = 0; i < D; ++i) {
      assert(v2[i] == -static_cast<T>(i + 1));
    }
  }
}

template <Size D, typename T>
HOSTDEV
TEST_CASE(compound_add)
{
  um2::Vec<D, T> v = makeVec<D, T>();
  um2::Vec<D, T> v2 = makeVec<D, T>();
  v += v2;
  if constexpr (std::floating_point<T>) {
    for (Size i = 0; i < D; ++i) {
      ASSERT_NEAR(v[i], static_cast<T>(2 * (i + 1)), static_cast<T>(1e-6));
    }
  } else {
    for (Size i = 0; i < D; ++i) {
      assert(v[i] == static_cast<T>(2 * (i + 1)));
    }
  }
}

template <Size D, typename T>
HOSTDEV
TEST_CASE(compound_sub)
{
  um2::Vec<D, T> v = makeVec<D, T>();
  um2::Vec<D, T> v2 = makeVec<D, T>();
  v -= v2;
  if constexpr (std::floating_point<T>) {
    for (Size i = 0; i < D; ++i) {
      ASSERT_NEAR(v[i], 0, static_cast<T>(1e-6));
    }
  } else {
    for (Size i = 0; i < D; ++i) {
      assert(v[i] == 0);
    }
  }
}

template <Size D, typename T>
HOSTDEV
TEST_CASE(compound_mul)
{
  um2::Vec<D, T> v = makeVec<D, T>();
  um2::Vec<D, T> v2 = makeVec<D, T>();
  v.array() *= v2.array();
  if constexpr (std::floating_point<T>) {
    for (Size i = 0; i < D; ++i) {
      ASSERT_NEAR(v[i], static_cast<T>((i + 1) * (i + 1)), static_cast<T>(1e-6));
    }
  } else {
    for (Size i = 0; i < D; ++i) {
      assert(v[i] == static_cast<T>((i + 1) * (i + 1)));
    }
  }
}

template <Size D, typename T>
HOSTDEV
TEST_CASE(compound_div)
{
  um2::Vec<D, T> v = makeVec<D, T>();
  um2::Vec<D, T> v2 = makeVec<D, T>();
  v.array() /= v2.array();
  if constexpr (std::floating_point<T>) {
    for (Size i = 0; i < D; ++i) {
      ASSERT_NEAR(v[i], 1, static_cast<T>(1e-6));
    }
  } else {
    for (Size i = 0; i < D; ++i) {
      assert(v[i] == 1);
    }
  }
}

template <Size D, typename T>
HOSTDEV
TEST_CASE(add)
{
  um2::Vec<D, T> v0 = makeVec<D, T>();
  um2::Vec<D, T> v1 = makeVec<D, T>();
  um2::Vec<D, T> v = v0 + v1;
  if constexpr (std::floating_point<T>) {
    for (Size i = 0; i < D; ++i) {
      ASSERT_NEAR(v[i], static_cast<T>(2 * (i + 1)), static_cast<T>(1e-6));
    }
  } else {
    for (Size i = 0; i < D; ++i) {
      assert(v[i] == static_cast<T>(2 * (i + 1)));
    }
  }
}

template <Size D, typename T>
HOSTDEV
TEST_CASE(sub)
{
  um2::Vec<D, T> v0 = makeVec<D, T>();
  um2::Vec<D, T> v1 = makeVec<D, T>();
  um2::Vec<D, T> v = v0 - v1;
  if constexpr (std::floating_point<T>) {
    for (Size i = 0; i < D; ++i) {
      ASSERT_NEAR(v[i], 0, static_cast<T>(1e-6));
    }
  } else {
    for (Size i = 0; i < D; ++i) {
      assert(v[i] == 0);
    }
  }
}

template <Size D, typename T>
HOSTDEV
TEST_CASE(mul)
{
  um2::Vec<D, T> v0 = makeVec<D, T>();
  um2::Vec<D, T> v1 = makeVec<D, T>();
  um2::Vec<D, T> v = v0.array() * v1.array();
  if constexpr (std::floating_point<T>) {
    for (Size i = 0; i < D; ++i) {
      ASSERT_NEAR(v[i], static_cast<T>((i + 1) * (i + 1)), static_cast<T>(1e-6));
    }
  } else {
    for (Size i = 0; i < D; ++i) {
      assert(v[i] == static_cast<T>((i + 1) * (i + 1)));
    }
  }
}

template <Size D, typename T>
HOSTDEV
TEST_CASE(div)
{
  um2::Vec<D, T> v0 = makeVec<D, T>();
  um2::Vec<D, T> v1 = makeVec<D, T>();
  um2::Vec<D, T> v = v0.array() / v1.array();
  if constexpr (std::floating_point<T>) {
    for (Size i = 0; i < D; ++i) {
      ASSERT_NEAR(v[i], 1, static_cast<T>(1e-6));
    }
  } else {
    for (Size i = 0; i < D; ++i) {
      assert(v[i] == 1);
    }
  }
}

template <Size D, typename T>
HOSTDEV
TEST_CASE(compound_scalar_add)
{
  um2::Vec<D, T> v = makeVec<D, T>();
  v.array() += 2;
  if constexpr (std::floating_point<T>) {
    for (Size i = 0; i < D; ++i) {
      ASSERT_NEAR(v[i], static_cast<T>(i + 3), static_cast<T>(1e-6));
    }
  } else {
    for (Size i = 0; i < D; ++i) {
      assert(v[i] == static_cast<T>(i + 3));
    }
  }
}

template <Size D, typename T>
HOSTDEV
TEST_CASE(compound_scalar_sub)
{
  um2::Vec<D, T> v = makeVec<D, T>();
  v.array() -= 2;
  if constexpr (std::floating_point<T>) {
    for (Size i = 0; i < D; ++i) {
      ASSERT_NEAR(v[i], static_cast<T>(i - 1), static_cast<T>(1e-6));
    }
  } else {
    for (Size i = 0; i < D; ++i) {
      assert(v[i] == static_cast<T>(i - 1));
    }
  }
}

template <Size D, typename T>
HOSTDEV
TEST_CASE(compound_scalar_mul)
{
  um2::Vec<D, T> v = makeVec<D, T>();
  v *= 2;
  if constexpr (std::floating_point<T>) {
    for (Size i = 0; i < D; ++i) {
      ASSERT_NEAR(v[i], static_cast<T>(2 * (i + 1)), static_cast<T>(1e-6));
    }
  } else {
    for (Size i = 0; i < D; ++i) {
      assert(v[i] == static_cast<T>(2 * (i + 1)));
    }
  }
}

template <Size D, typename T>
HOSTDEV
TEST_CASE(compound_scalar_div)
{
  um2::Vec<D, T> v = makeVec<D, T>();
  v /= 2;
  if constexpr (std::floating_point<T>) {
    for (Size i = 0; i < D; ++i) {
      ASSERT_NEAR(v[i], static_cast<T>(i + 1) / 2, static_cast<T>(1e-6));
    }
  } else {
    for (Size i = 0; i < D; ++i) {
      assert(v[i] == static_cast<T>(i + 1) / 2);
    }
  }
}

template <Size D, typename T>
HOSTDEV
TEST_CASE(scalar_add)
{
  um2::Vec<D, T> v0 = makeVec<D, T>();
  um2::Vec<D, T> v = v0.array() + 2;
  if constexpr (std::floating_point<T>) {
    for (Size i = 0; i < D; ++i) {
      ASSERT_NEAR(v[i], static_cast<T>(i + 3), static_cast<T>(1e-6));
    }
  } else {
    for (Size i = 0; i < D; ++i) {
      assert(v[i] == static_cast<T>(i + 3));
    }
  }
}

template <Size D, typename T>
HOSTDEV
TEST_CASE(scalar_sub)
{
  um2::Vec<D, T> v0 = makeVec<D, T>();
  um2::Vec<D, T> v = v0.array() - 2;
  if constexpr (std::floating_point<T>) {
    for (Size i = 0; i < D; ++i) {
      ASSERT_NEAR(v[i], static_cast<T>(i - 1), static_cast<T>(1e-6));
    }
  } else {
    for (Size i = 0; i < D; ++i) {
      assert(v[i] == static_cast<T>(i - 1));
    }
  }
}

template <Size D, typename T>
HOSTDEV
TEST_CASE(scalar_mul)
{
  um2::Vec<D, T> v0 = makeVec<D, T>();
  um2::Vec<D, T> v = v0 * 2;
  if constexpr (std::floating_point<T>) {
    for (Size i = 0; i < D; ++i) {
      ASSERT_NEAR(v[i], static_cast<T>(2 * (i + 1)), static_cast<T>(1e-6));
    }
  } else {
    for (Size i = 0; i < D; ++i) {
      assert(v[i] == static_cast<T>(2 * (i + 1)));
    }
  }
}

template <Size D, typename T>
HOSTDEV
TEST_CASE(scalar_div)
{
  um2::Vec<D, T> v0 = makeVec<D, T>();
  um2::Vec<D, T> v = v0 / 2;
  if constexpr (std::floating_point<T>) {
    for (Size i = 0; i < D; ++i) {
      ASSERT_NEAR(v[i], static_cast<T>(i + 1) / 2, static_cast<T>(1e-6));
    }
  } else {
    for (Size i = 0; i < D; ++i) {
      assert(v[i] == static_cast<T>(i + 1) / 2);
    }
  }
}

template <Size D, typename T>
HOSTDEV
TEST_CASE(min)
{
  um2::Vec<D, T> v0 = makeVec<D, T>();
  um2::Vec<D, T> v1 = makeVec<D, T>().array() + 1;
  um2::Vec<D, T> v = v0.cwiseMin(v1);
  if constexpr (std::floating_point<T>) {
    for (Size i = 0; i < D; ++i) {
      ASSERT_NEAR(v[i], static_cast<T>(i + 1), static_cast<T>(1e-6));
    }
  } else {
    for (Size i = 0; i < D; ++i) {
      assert(v[i] == static_cast<T>(i + 1));
    }
  }
}

template <Size D, typename T>
HOSTDEV
TEST_CASE(max)
{
  um2::Vec<D, T> v0 = makeVec<D, T>();
  um2::Vec<D, T> v1 = makeVec<D, T>().array() + 1;
  um2::Vec<D, T> v = v0.cwiseMax(v1);
  if constexpr (std::floating_point<T>) {
    for (Size i = 0; i < D; ++i) {
      ASSERT_NEAR(v[i], static_cast<T>(i + 2), static_cast<T>(1e-6));
    }
  } else {
    for (Size i = 0; i < D; ++i) {
      assert(v[i] == static_cast<T>(i + 2));
    }
  }
}

template <Size D, typename T>
HOSTDEV
TEST_CASE(dot)
{
  um2::Vec<D, T> v = makeVec<D, T>();
  T dot = v.dot(v);
  if constexpr (std::floating_point<T>) {
    ASSERT_NEAR(dot, static_cast<T>(D * (D + 1) * (2 * D + 1)) / 6, static_cast<T>(1e-6));
  } else {
    assert(dot == static_cast<T>(D * (D + 1) * (2 * D + 1) / 6));
  }
}

template <Size D, typename T>
HOSTDEV
TEST_CASE(cross)
{
  if constexpr (D == 3) {
    um2::Vec<D, T> v0 = makeVec<D, T>();
    um2::Vec<D, T> v1 = makeVec<D, T>().array() + 1;
    um2::Vec<D, T> v = v0.cross(v1);
    ASSERT_NEAR(v[0], -1, static_cast<T>(1e-6));
    ASSERT_NEAR(v[1], 2, static_cast<T>(1e-6));
    ASSERT_NEAR(v[2], -1, static_cast<T>(1e-6));
    v *= 0;
  } else if constexpr (D == 2) {
    um2::Vec2<T> v0(1, 2);
    um2::Vec2<T> v1(3, 4);
    T x = um2::cross2(v0, v1);
    ASSERT_NEAR(x, -2, static_cast<T>(1e-6));
  }
}

template <Size D, typename T>
HOSTDEV
TEST_CASE(norm2)
{
  um2::Vec<D, T> v = makeVec<D, T>();
  T norm2 = v.squaredNorm();
  if constexpr (std::floating_point<T>) {
    ASSERT_NEAR(norm2, static_cast<T>(D * (D + 1) * (2 * D + 1)) / 6,
                static_cast<T>(1e-6));
  } else {
    assert(norm2 == static_cast<T>(D * (D + 1) * (2 * D + 1) / 6));
  }
}

template <Size D, typename T>
HOSTDEV
TEST_CASE(norm)
{
  um2::Vec<D, T> v = makeVec<D, T>();
  T norm = v.norm();
  ASSERT_NEAR(norm, std::sqrt(static_cast<T>(D * (D + 1) * (2 * D + 1)) / 6),
              static_cast<T>(1e-6));
}

template <Size D, typename T>
HOSTDEV
TEST_CASE(normalize)
{
  um2::Vec<D, T> v = makeVec<D, T>();
  um2::Vec<D, T> v2 = v.normalized();
  T n = v2.norm();
  ASSERT_NEAR(n, 1, static_cast<T>(1e-6));
}
// --------------------------------------------------------------------------
// CUDA
// --------------------------------------------------------------------------
#if UM2_ENABLE_CUDA
template <Size D, typename T>
MAKE_CUDA_KERNEL(accessor, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(unary_minus, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(compound_add, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(compound_sub, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(compound_mul, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(compound_div, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(add, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(sub, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(mul, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(div, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(compound_scalar_add, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(compound_scalar_sub, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(compound_scalar_mul, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(compound_scalar_div, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(scalar_add, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(scalar_sub, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(scalar_mul, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(scalar_div, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(min, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(max, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(dot, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(cross, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(norm2, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(norm, D, T);

template <Size D, typename T>
MAKE_CUDA_KERNEL(normalize, D, T);

#endif

template <Size D, typename T>
TEST_SUITE(vec)
{
  TEST_HOSTDEV(accessor, 1, 1, D, T);
  if constexpr (!std::unsigned_integral<T>) {
    TEST_HOSTDEV(unary_minus, 1, 1, D, T);
  }
  TEST_HOSTDEV(compound_add, 1, 1, D, T);
  TEST_HOSTDEV(compound_sub, 1, 1, D, T);
  TEST_HOSTDEV(compound_mul, 1, 1, D, T);
  TEST_HOSTDEV(compound_div, 1, 1, D, T);
  TEST_HOSTDEV(add, 1, 1, D, T);
  TEST_HOSTDEV(sub, 1, 1, D, T);
  TEST_HOSTDEV(mul, 1, 1, D, T);
  TEST_HOSTDEV(div, 1, 1, D, T);
  TEST_HOSTDEV(compound_scalar_add, 1, 1, D, T);
  TEST_HOSTDEV(compound_scalar_sub, 1, 1, D, T);
  TEST_HOSTDEV(compound_scalar_mul, 1, 1, D, T);
  TEST_HOSTDEV(compound_scalar_div, 1, 1, D, T);
  TEST_HOSTDEV(scalar_add, 1, 1, D, T);
  TEST_HOSTDEV(scalar_sub, 1, 1, D, T);
  TEST_HOSTDEV(scalar_mul, 1, 1, D, T);
  TEST_HOSTDEV(scalar_div, 1, 1, D, T);
  TEST_HOSTDEV(min, 1, 1, D, T);
  TEST_HOSTDEV(max, 1, 1, D, T);
  TEST_HOSTDEV(dot, 1, 1, D, T);
  TEST_HOSTDEV(norm2, 1, 1, D, T);
  if constexpr (std::floating_point<T>) {
    TEST_HOSTDEV(cross, 1, 1, D, T);
    TEST_HOSTDEV(norm, 1, 1, D, T);
    TEST_HOSTDEV(normalize, 1, 1, D, T);
  }
}

auto
main() -> int
{
  RUN_SUITE((vec<2, float>));
  RUN_SUITE((vec<2, double>));
  RUN_SUITE((vec<2, int32_t>));
  RUN_SUITE((vec<2, uint32_t>));

  RUN_SUITE((vec<3, float>));
  RUN_SUITE((vec<3, double>));
  RUN_SUITE((vec<3, int32_t>));
  RUN_SUITE((vec<3, uint32_t>));

  RUN_SUITE((vec<4, float>));
  RUN_SUITE((vec<4, double>));
  RUN_SUITE((vec<4, int32_t>));
  RUN_SUITE((vec<4, uint32_t>));
  return 0;
}
