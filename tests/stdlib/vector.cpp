#include <um2/config.hpp>
#include <um2/stdlib/vector.hpp>

#include <concepts> // std::floating_point
#include <cstdint>
#include <type_traits>

#include "../test_macros.hpp"

//==============================================================================
// Constructors
//==============================================================================

template <class T>
HOSTDEV
TEST_CASE(constructor_copy)
{
  um2::Vector<T> v(10);
  for (Int i = 0; i < 10; i++) {
    v[i] = static_cast<T>(i);
  }
  um2::Vector<T> v2(v);
  ASSERT(v2.size() == 10);
  ASSERT(v2.capacity() == 10);
  for (Int i = 0; i < 10; i++) {
    if constexpr (std::floating_point<T>) {
      ASSERT_NEAR(v2[i], static_cast<T>(i), static_cast<T>(1e-6));
    } else {
      ASSERT(v2[i] == static_cast<T>(i));
    }
  }
}

template <class T>
HOSTDEV auto
createVector(Int size) -> um2::Vector<T>
{
  um2::Vector<T> v(size);
  for (Int i = 0; i < size; i++) {
    v[i] = static_cast<T>(i);
  }
  return v;
}

template <class T>
HOSTDEV
TEST_CASE(constructor_move)
{
  um2::Vector<T> const v(um2::move(createVector<T>(10)));
  ASSERT(v.cbegin() != nullptr);
  ASSERT(v.cend() != nullptr);
  ASSERT(v.size() == 10);
  ASSERT(v.capacity() == 10);
  for (Int i = 0; i < 10; i++) {
    if constexpr (std::floating_point<T>) {
      ASSERT_NEAR(v[i], static_cast<T>(i), static_cast<T>(1e-6));
    } else {
      ASSERT(v[i] == static_cast<T>(i));
    }
  }
}

template <class T>
HOSTDEV
TEST_CASE(constructor_size)
{
  um2::Vector<T> const v(10);
  ASSERT(v.cbegin() != nullptr);
  ASSERT(v.cend() != nullptr);
  ASSERT(v.size() == 10);
  ASSERT(v.capacity() == 10);
  for (Int i = 0; i < 10; i++) {
    if constexpr (std::floating_point<T>) {
      ASSERT_NEAR(v[i], static_cast<T>(0), static_cast<T>(1e-6));
    } else {
      ASSERT(v[i] == 0);
    }
  }
}

template <class T>
HOSTDEV
TEST_CASE(constructor_size_value)
{
  um2::Vector<T> v(10, 2);
  ASSERT(v.cbegin() != nullptr);
  ASSERT(v.cend() != nullptr);
  ASSERT(v.size() == 10);
  ASSERT(v.capacity() == 10);
  for (Int i = 0; i < 10; ++i) {
    if constexpr (std::floating_point<T>) {
      ASSERT_NEAR(v[i], static_cast<T>(2), static_cast<T>(1e-6));
    } else {
      ASSERT(v[i] == 2);
    }
  }
}

template <class T>
TEST_CASE(constructor_initializer_list)
{
  um2::Vector<T> v{1, 2, 3, 4, 5};
  ASSERT(v.size() == 5);
  ASSERT(v.capacity() == 5);
  for (Int i = 0; i < 5; ++i) {
    if constexpr (std::floating_point<T>) {
      ASSERT_NEAR(v[i], static_cast<T>(i + 1), static_cast<T>(1e-6));
    } else {
      ASSERT(v[i] == static_cast<T>(i + 1));
    }
  }
}

template <class T>
HOSTDEV
TEST_CASE(assign_copy)
{
  um2::Vector<T> v(10);
  for (Int i = 0; i < 10; i++) {
    v[i] = static_cast<T>(i);
  }

  um2::Vector<T> v2(6);
  v2 = v;
  ASSERT(v2.size() == 10);
  ASSERT(v2.capacity() == 10);
  for (Int i = 0; i < 10; i++) {
    if constexpr (std::floating_point<T>) {
      ASSERT_NEAR(v2[i], static_cast<T>(i), static_cast<T>(1e-6));
    } else {
      ASSERT(v2[i] == static_cast<T>(i));
    }
  }

  um2::Vector<T> v3(33);
  v3 = v;
  ASSERT(v3.size() == 10);
  ASSERT(v3.capacity() == 33);
  for (Int i = 0; i < 10; i++) {
    if constexpr (std::floating_point<T>) {
      ASSERT_NEAR(v3[i], static_cast<T>(i), static_cast<T>(1e-6));
    } else {
      ASSERT(v3[i] == static_cast<T>(i));
    }
  }
}

template <class T>
HOSTDEV
TEST_CASE(assign_move)
{
  um2::Vector<T> v1;
  v1 = um2::move(createVector<T>(10));
  ASSERT(v1.cbegin() != nullptr);
  ASSERT(v1.cend() != nullptr);
  ASSERT(v1.size() == 10);
  ASSERT(v1.capacity() == 10);
}

template <class T>
HOSTDEV
TEST_CASE(assign_initializer_list)
{
  um2::Vector<T> v = {1, 2, 3, 4, 5};
  ASSERT(v.size() == 5);
  ASSERT(v.capacity() == 5);
  for (Int i = 0; i < 5; ++i) {
    if constexpr (std::floating_point<T>) {
      ASSERT_NEAR(v[i], static_cast<T>(i + 1), static_cast<T>(1e-6));
    } else {
      ASSERT(v[i] == static_cast<T>(i + 1));
    }
  }
}

//==============================================================================
// Relational operators
//==============================================================================

HOSTDEV
TEST_CASE(relational_operators)
{
  um2::Vector<int> v1(3);
  um2::Vector<int> v2(3);
  um2::Vector<int> v3(3);
  for (Int i = 0; i < 3; ++i) {
    v1[i] = i;
    v2[i] = i;
    v3[i] = i;
  }
  v3[2] = 5;
  ASSERT(v1 == v2);
  ASSERT(v1 <= v2);
  ASSERT(v1 <= v3);
  ASSERT(v3 >= v2);
  ASSERT(v3 > v2);
  ASSERT(v1 < v3);
  ASSERT(v1 != v3);
  v2.push_back(4);
  ASSERT(v1 != v2);
  ASSERT(v1 <= v2);
  ASSERT(v3 >= v2);
  ASSERT(v3 > v2);
}

//==============================================================================--
// Modifiers
//==============================================================================--

// NOLINTBEGIN(cppcoreguidelines-avoid-non-const-global-variables) OK
#ifndef __CUDA_ARCH__
int count = 0;
#else
DEVICE int count = 0;
#endif
// NOLINTEND(cppcoreguidelines-avoid-non-const-global-variables)

struct Counted {
  int my_count = 0;

  HOSTDEV
  Counted()
      : my_count(++count){};

  HOSTDEV ~Counted() { --count; }

  // Delete the move and copy constructors and operators.
  Counted(Counted &&) = delete;
  Counted(const Counted &) = delete;

  auto
  operator=(Counted &&) -> Counted & = delete;

  auto
  operator=(const Counted &) -> Counted & = delete;
};

HOSTDEV
TEST_CASE(clear)
{
  count = 0;
  um2::Vector<Counted> v(10);
  ASSERT(count == 10);
  v.clear();
  ASSERT(count == 0);
  ASSERT(v.empty());
  ASSERT(v.capacity() == 10);
}

template <class T>
HOSTDEV
TEST_CASE(resize)
{
  um2::Vector<T> v;
  v.resize(0);
  ASSERT(v.empty());
  ASSERT(v.capacity() == 0);
  ASSERT(v.data() == nullptr);
  v.resize(1);
  ASSERT(v.size() == 1);
  ASSERT(v.capacity() == 1);
  ASSERT(v.data() != nullptr);
  v.resize(2);
  ASSERT(v.size() == 2);
  ASSERT(v.capacity() == 2);
  ASSERT(v.data() != nullptr);
}

template <class T>
HOSTDEV
TEST_CASE(reserve)
{
  um2::Vector<T> v;
  v.reserve(1);
  ASSERT(v.empty());
  ASSERT(v.capacity() == 1);
  ASSERT(v.data() != nullptr);
  v.reserve(2);
  v.push_back(static_cast<T>(1));
  ASSERT(v.size() == 1);
  ASSERT(v.capacity() == 2);
  ASSERT(v.data() != nullptr);
  v.reserve(5);
  v.push_back(static_cast<T>(1));
  ASSERT(v.size() == 2);
  ASSERT(v.capacity() == 5);
  ASSERT(v.data() != nullptr);
  v.reserve(7);
  ASSERT(v.size() == 2);
  ASSERT(v.capacity() == 7);
  if constexpr (std::floating_point<T>) {
    ASSERT_NEAR(v[0], static_cast<T>(1), static_cast<T>(1e-6));
    ASSERT_NEAR(v[1], static_cast<T>(1), static_cast<T>(1e-6));
  } else {
    ASSERT(v[0] == static_cast<T>(1));
    ASSERT(v[1] == static_cast<T>(1));
  }
}

template <class T>
HOSTDEV
TEST_CASE(push_back)
{
  um2::Vector<T> v;
  v.push_back(1);
  ASSERT(v.size() == 1);
  ASSERT(v.capacity() == 1);
  if constexpr (std::floating_point<T>) {
    ASSERT_NEAR(v.data()[0], static_cast<T>(1), static_cast<T>(1e-6));
  } else {
    ASSERT(v.data()[0] == static_cast<T>(1));
  }
  v.push_back(2);
  ASSERT(v.size() == 2);
  ASSERT(v.capacity() == 2);
  if constexpr (std::floating_point<T>) {
    ASSERT_NEAR(v.data()[0], static_cast<T>(1), static_cast<T>(1e-6));
    ASSERT_NEAR(v.data()[1], static_cast<T>(2), static_cast<T>(1e-6));
  } else {
    ASSERT(v.data()[0] == static_cast<T>(1));
    ASSERT(v.data()[1] == static_cast<T>(2));
  }
  v.push_back(3);
  ASSERT(v.size() == 3);
  ASSERT(v.capacity() == 4);
  if constexpr (std::floating_point<T>) {
    ASSERT_NEAR(v.data()[0], static_cast<T>(1), static_cast<T>(1e-6));
    ASSERT_NEAR(v.data()[1], static_cast<T>(2), static_cast<T>(1e-6));
    ASSERT_NEAR(v.data()[2], static_cast<T>(3), static_cast<T>(1e-6));
  } else {
    ASSERT(v.data()[0] == static_cast<T>(1));
    ASSERT(v.data()[1] == static_cast<T>(2));
    ASSERT(v.data()[2] == static_cast<T>(3));
  }
  v.clear();
  v.push_back(1);
  v.push_back(2);
  v.push_back(3);
  v.push_back(4);
  v.push_back(5);
  ASSERT(v.size() == 5);
  ASSERT(v.capacity() == 8);
  if constexpr (std::floating_point<T>) {
    ASSERT_NEAR(v.data()[0], static_cast<T>(1), static_cast<T>(1e-6));
    ASSERT_NEAR(v.data()[1], static_cast<T>(2), static_cast<T>(1e-6));
    ASSERT_NEAR(v.data()[2], static_cast<T>(3), static_cast<T>(1e-6));
    ASSERT_NEAR(v.data()[3], static_cast<T>(4), static_cast<T>(1e-6));
    ASSERT_NEAR(v.data()[4], static_cast<T>(5), static_cast<T>(1e-6));
  } else {
    ASSERT(v.data()[0] == static_cast<T>(1));
    ASSERT(v.data()[1] == static_cast<T>(2));
    ASSERT(v.data()[2] == static_cast<T>(3));
    ASSERT(v.data()[3] == static_cast<T>(4));
    ASSERT(v.data()[4] == static_cast<T>(5));
  }
}

template <class T>
HOSTDEV
TEST_CASE(push_back_rval_ref)
{
  um2::Vector<T> l_value_vector;
  l_value_vector.push_back(static_cast<T>(1));
  l_value_vector.push_back(static_cast<T>(2));
  um2::Vector<um2::Vector<T>> my_vector;
  my_vector.push_back(l_value_vector);
  my_vector.push_back(std::move(l_value_vector));
  ASSERT(my_vector.size() == 2);
  ASSERT(my_vector.capacity() == 2);
  if constexpr (std::floating_point<T>) {
    ASSERT_NEAR(my_vector[0][0], static_cast<T>(1), static_cast<T>(1e-6));
    ASSERT_NEAR(my_vector[1][0], static_cast<T>(1), static_cast<T>(1e-6));
  } else {
    ASSERT(my_vector[0][0] == static_cast<T>(1));
    ASSERT(my_vector[1][0] == static_cast<T>(1));
  }
}

HOSTDEV
TEST_CASE(emplace_back)
{
  static_assert(!std::is_trivially_move_constructible_v<um2::Vector<int>>);
  static_assert(!std::is_trivially_destructible_v<um2::Vector<int>>);
  static_assert(std::is_trivially_move_constructible_v<int>);
  static_assert(std::is_trivially_destructible_v<int>);
  struct TestStruct {
    int a;
    float b;
    double c;

    HOSTDEV
    TestStruct(int ia, float ib, double ic)
        : a(ia),
          b(ib),
          c(ic)
    {
    }
  };
  um2::Vector<TestStruct> v;
  v.emplace_back(1, 2.0F, 3.0);
  ASSERT(v.size() == 1);
  ASSERT(v.capacity() == 1);
  ASSERT(v[0].a == 1);

  um2::Vector<int> v2 = {1, 2, 3};
  um2::Vector<um2::Vector<int>> v3;
  v3.emplace_back(v2);
  v3.emplace_back(std::move(v2));
  ASSERT(v3.size() == 2);
  ASSERT(v3.capacity() == 2);
  for (Int i = 0; i < 2; ++i) {
    ASSERT(v3[i].size() == 3);
    for (Int j = 0; j < 3; ++j) {
      ASSERT(v3[i][j] == j + 1);
    }
  }
}

////==============================================================================
//// CUDA
////==============================================================================
//
// #if UM2_USE_CUDA
// template <class T>
// MAKE_CUDA_KERNEL(constructor_Size, T)
//
// template <class T>
// MAKE_CUDA_KERNEL(constructor_Size_value, T)
//
// template <class T>
// MAKE_CUDA_KERNEL(copy_constructor, T)
//
// template <class T>
// MAKE_CUDA_KERNEL(move_constructor, T)
//
// template <class T>
// MAKE_CUDA_KERNEL(constructor_initializer_list, T)
//
// template <class T>
// MAKE_CUDA_KERNEL(operator_copy, T)
//
// template <class T>
// MAKE_CUDA_KERNEL(operator_move, T)
//
// template <class T>
// MAKE_CUDA_KERNEL(resize, T)
//
// template <class T>
// MAKE_CUDA_KERNEL(reserve, T)
//
// template <class T>
// MAKE_CUDA_KERNEL(push_back, T)
//
// template <class T>
// MAKE_CUDA_KERNEL(push_back_rval_ref, T)
//
// template <class T>
// MAKE_CUDA_KERNEL(push_back_n, T)
//
// MAKE_CUDA_KERNEL(clear)
//
//    MAKE_CUDA_KERNEL(emplace_back)
// #endif // UM2_USE_CUDA
//
template <class T>
TEST_SUITE(Vector)
{
  // Constructors and assignment
  TEST_HOSTDEV(constructor_copy, T)
  TEST_HOSTDEV(constructor_move, T)
  TEST_HOSTDEV(constructor_size, T)
  TEST_HOSTDEV(constructor_size_value, T)
  TEST(constructor_initializer_list<T>)
  TEST_HOSTDEV(assign_copy, T)
  TEST_HOSTDEV(assign_move, T)
  TEST(assign_initializer_list<T>)

  // Operators
  TEST_HOSTDEV(relational_operators)

  // Modifiers
  TEST_HOSTDEV(clear)
  TEST_HOSTDEV(resize, T)
  TEST_HOSTDEV(reserve, T)
  TEST_HOSTDEV(push_back, T)
  TEST_HOSTDEV(push_back_rval_ref, T)
  TEST_HOSTDEV(emplace_back)
}

auto
main() -> int
{
  RUN_SUITE(Vector<int32_t>);
  RUN_SUITE(Vector<uint32_t>);
  RUN_SUITE(Vector<int64_t>);
  RUN_SUITE(Vector<uint64_t>);
  RUN_SUITE(Vector<float>);
  RUN_SUITE(Vector<double>);
  return 0;
}
