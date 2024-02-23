#include <um2/stdlib/vector.hpp>

#include <concepts> // std::floating_point
#include <vector> 

#include <type_traits>

#include "../test_macros.hpp"

#define CHECK_STD_VECTOR 1

//==============================================================================
// Constructors
//==============================================================================

template <class T>
HOSTDEV
TEST_CASE(constructor_n)
{
  um2::Vector<T> const v(10);
  ASSERT(v.cbegin() != nullptr);
  ASSERT(v.cend() != nullptr);
  ASSERT(v.size() == 10);
  ASSERT(v.capacity() == 10);

#if CHECK_STD_VECTOR
  std::vector<T> const stdv(10);
  ASSERT(stdv.size() == 10);
  ASSERT(stdv.capacity() == 10);
#endif
}

template <class T>
HOSTDEV
TEST_CASE(constructor_n_value)
{
  um2::Vector<T> v(10, 2);
  ASSERT(v.cbegin() != nullptr);
  ASSERT(v.cend() != nullptr);
  ASSERT(v.size() == 10);
  ASSERT(v.capacity() == 10);
  for (int i = 0; i < 10; ++i) {
    if constexpr (std::floating_point<T>) {
      ASSERT_NEAR(v[i], static_cast<T>(2), static_cast<T>(1e-6));
    } else {
      ASSERT(v[i] == 2);
    }
  }

#if CHECK_STD_VECTOR
  std::vector<T> stdv(10, 2);
  ASSERT(stdv.size() == 10);
  ASSERT(stdv.capacity() == 10);
  for (size_t i = 0; i < 10; ++i) {
    if constexpr (std::floating_point<T>) {
      ASSERT_NEAR(stdv[i], static_cast<T>(2), static_cast<T>(1e-6));
    } else {
      ASSERT(stdv[i] == 2);
    }
  }
#endif
}

template <class T>
HOSTDEV
TEST_CASE(copy_constructor)
{
  um2::Vector<T> v(10);
  for (int i = 0; i < 10; i++) {
    v[i] = static_cast<T>(i);
  }
  um2::Vector<T> v2(v);
  ASSERT(v2.size() == 10);
  ASSERT(v2.capacity() == 10);
  for (int i = 0; i < 10; i++) {
    if constexpr (std::floating_point<T>) {
      ASSERT_NEAR(v2[i], static_cast<T>(i), static_cast<T>(1e-6));
    } else {
      ASSERT(v2[i] == static_cast<T>(i));
    }
  }

#if CHECK_STD_VECTOR
  std::vector<T> stdv(10);
  for (size_t i = 0; i < 10; i++) {
    stdv[i] = static_cast<T>(i);
  }
  std::vector<T> stdv2(stdv);
  ASSERT(stdv2.size() == 10);
  ASSERT(stdv2.capacity() == 10);
  for (size_t i = 0; i < 10; i++) {
    if constexpr (std::floating_point<T>) {
      ASSERT_NEAR(stdv2[i], static_cast<T>(i), static_cast<T>(1e-6));
    } else {
      ASSERT(stdv2[i] == static_cast<T>(i));
    }
  }
#endif
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


#if CHECK_STD_VECTOR
template <class T>
auto
// NOLINTNEXTLINE(readability-identifier-naming)
create_std_vector(Int size) -> std::vector<T>
{
  auto const n = static_cast<size_t>(size);
  std::vector<T> v(n);
  for (size_t i = 0; i < n; i++) {
    v[i] = static_cast<T>(i);
  }
  return v;
}
#endif

template <class T>
HOSTDEV
TEST_CASE(move_constructor)
{
  um2::Vector<T> const v(um2::move(createVector<T>(10)));
  ASSERT(v.cbegin() != nullptr);
  ASSERT(v.cend() != nullptr);
  ASSERT(v.size() == 10);
  ASSERT(v.capacity() == 10);

#if CHECK_STD_VECTOR
  std::vector<T> const stdv(um2::move(create_std_vector<T>(10)));
  ASSERT(stdv.size() == 10);
  ASSERT(stdv.capacity() == 10);
#endif
}

template <class T>
HOSTDEV
TEST_CASE(constructor_initializer_list)
{
  um2::Vector<T> v{1, 2, 3, 4, 5};
  ASSERT(v.size() == 5);
  ASSERT(v.capacity() == 5);
  for (int i = 0; i < 5; ++i) {
    if constexpr (std::floating_point<T>) {
      ASSERT_NEAR(v[i], static_cast<T>(i + 1), static_cast<T>(1e-6));
    } else {
      ASSERT(v[i] == static_cast<T>(i + 1));
    }
  }
}

//==============================================================================
// Operators
//==============================================================================

template <class T>
HOSTDEV
TEST_CASE(operator_copy)
{
  um2::Vector<T> v(10);
  for (int i = 0; i < 10; i++) {
    v[i] = static_cast<T>(i);
  }

  um2::Vector<T> v2(6);
  v2 = v;
  ASSERT(v2.size() == 10);
  ASSERT(v2.capacity() == 10);
  for (int i = 0; i < 10; i++) {
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
  for (int i = 0; i < 10; i++) {
    if constexpr (std::floating_point<T>) {
      ASSERT_NEAR(v3[i], static_cast<T>(i), static_cast<T>(1e-6));
    } else {
      ASSERT(v3[i] == static_cast<T>(i));
    }
  }

#if CHECK_STD_VECTOR
  std::vector<T> stdv(10);
  for (size_t i = 0; i < 10; i++) {
    stdv[i] = static_cast<T>(i);
  }

  std::vector<T> stdv2(6);
  stdv2 = stdv;
  ASSERT(stdv2.size() == 10);
  ASSERT(stdv2.capacity() == 10);
  for (size_t i = 0; i < 10; i++) {
    if constexpr (std::floating_point<T>) {
      ASSERT_NEAR(stdv2[i], static_cast<T>(i), static_cast<T>(1e-6));
    } else {
      ASSERT(stdv2[i] == static_cast<T>(i));
    }
  }

  std::vector<T> stdv3(33);
  stdv3 = stdv;
  ASSERT(stdv3.size() == 10);
  ASSERT(stdv3.capacity() == 33);
  for (size_t i = 0; i < 10; i++) {
    if constexpr (std::floating_point<T>) {
      ASSERT_NEAR(stdv3[i], static_cast<T>(i), static_cast<T>(1e-6));
    } else {
      ASSERT(stdv3[i] == static_cast<T>(i));
    }
  }
#endif
}

template <class T>
HOSTDEV
TEST_CASE(operator_move)
{
  um2::Vector<T> v1;
  v1 = um2::move(createVector<T>(10));
  ASSERT(v1.cbegin() != nullptr);
  ASSERT(v1.cend() != nullptr);
  ASSERT(v1.size() == 10);
  ASSERT(v1.capacity() == 10);

#if CHECK_STD_VECTOR
  std::vector<T> stdv1;
  stdv1 = um2::move(create_std_vector<T>(10));
  ASSERT(stdv1.size() == 10);
  ASSERT(stdv1.capacity() == 10);
#endif
}

template <class T>
HOSTDEV
TEST_CASE(operator_initializer_list)
{
  um2::Vector<T> v = {1, 2, 3, 4, 5};
  ASSERT(v.size() == 5);
  ASSERT(v.capacity() == 5);
  for (int i = 0; i < 5; ++i) {
    if constexpr (std::floating_point<T>) {
      ASSERT_NEAR(v[i], static_cast<T>(i + 1), static_cast<T>(1e-6));
    } else {
      ASSERT(v[i] == static_cast<T>(i + 1));
    }
  }
}

HOSTDEV
TEST_CASE(operator_equal)
{
  um2::Vector<int> const v1 = {1, 2, 3, 4, 5};
  um2::Vector<int> const v2 = {1, 2, 3, 4, 5};
  um2::Vector<int> const v3 = {1, 2, 3, 4, 6};
  ASSERT(v1 == v2);
  ASSERT(v1 != v3);
}

//==============================================================================--
// Functions 
//==============================================================================--

// NOLINTBEGIN; justification: Just a quick test struct
#ifndef __CUDA_ARCH__
int count = 0;
#else
DEVICE int count = 0;
#endif
struct Counted {
  HOSTDEV
  Counted() { ++count; }
  HOSTDEV
  Counted(Counted const &) { ++count; }
  HOSTDEV ~Counted() { --count; }
  HOSTDEV friend void operator&(Counted) = delete;
};
// NOLINTEND

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

#if CHECK_STD_VECTOR
  std::vector<T> stdv;
  stdv.resize(0);
  ASSERT(stdv.empty());
  ASSERT(stdv.capacity() == 0);
  stdv.resize(1);
  ASSERT(stdv.size() == 1);
  ASSERT(stdv.capacity() == 1);
  stdv.resize(2);
  ASSERT(stdv.size() == 2);
  ASSERT(stdv.capacity() == 2);
#endif
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

#if CHECK_STD_VECTOR
  std::vector<T> stdv;
  stdv.reserve(1);
  ASSERT(stdv.empty());
  ASSERT(stdv.capacity() == 1);
  stdv.reserve(2);
  stdv.push_back(static_cast<T>(1));
  ASSERT(stdv.size() == 1);
  ASSERT(stdv.capacity() == 2);
  stdv.reserve(5);
  stdv.push_back(static_cast<T>(1));
  ASSERT(stdv.size() == 2);
  ASSERT(stdv.capacity() == 5);
  stdv.reserve(7);
  ASSERT(stdv.size() == 2);
  ASSERT(stdv.capacity() == 7);
  if constexpr (std::floating_point<T>) {
    ASSERT_NEAR(stdv[0], static_cast<T>(1), static_cast<T>(1e-6));
    ASSERT_NEAR(stdv[1], static_cast<T>(1), static_cast<T>(1e-6));
  } else {
    ASSERT(stdv[0] == static_cast<T>(1));
    ASSERT(stdv[1] == static_cast<T>(1));
  }
#endif
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

#if CHECK_STD_VECTOR
  std::vector<T> stdv;
  stdv.push_back(1);
  ASSERT(stdv.size() == 1);
  ASSERT(stdv.capacity() == 1);
  if constexpr (std::floating_point<T>) {
    ASSERT_NEAR(stdv[0], static_cast<T>(1), static_cast<T>(1e-6));
  } else {
    ASSERT(stdv[0] == static_cast<T>(1));
  }
  stdv.push_back(2);
  ASSERT(stdv.size() == 2);
  ASSERT(stdv.capacity() == 2);
  if constexpr (std::floating_point<T>) {
    ASSERT_NEAR(stdv[0], static_cast<T>(1), static_cast<T>(1e-6));
    ASSERT_NEAR(stdv[1], static_cast<T>(2), static_cast<T>(1e-6));
  } else {
    ASSERT(stdv[0] == static_cast<T>(1));
    ASSERT(stdv[1] == static_cast<T>(2));
  }
  stdv.push_back(3);
  ASSERT(stdv.size() == 3);
  ASSERT(stdv.capacity() == 4);
  if constexpr (std::floating_point<T>) {
    ASSERT_NEAR(stdv[0], static_cast<T>(1), static_cast<T>(1e-6));
    ASSERT_NEAR(stdv[1], static_cast<T>(2), static_cast<T>(1e-6));
    ASSERT_NEAR(stdv[2], static_cast<T>(3), static_cast<T>(1e-6));
  } else {
    ASSERT(stdv[0] == static_cast<T>(1));
    ASSERT(stdv[1] == static_cast<T>(2));
    ASSERT(stdv[2] == static_cast<T>(3));
  }
  stdv.clear();
  stdv.push_back(1);
  stdv.push_back(2);
  stdv.push_back(3);
  stdv.push_back(4);
  stdv.push_back(5);
  ASSERT(stdv.size() == 5);
  ASSERT(stdv.capacity() == 8);
  if constexpr (std::floating_point<T>) {
    ASSERT_NEAR(stdv[0], static_cast<T>(1), static_cast<T>(1e-6));
    ASSERT_NEAR(stdv[1], static_cast<T>(2), static_cast<T>(1e-6));
    ASSERT_NEAR(stdv[2], static_cast<T>(3), static_cast<T>(1e-6));
    ASSERT_NEAR(stdv[3], static_cast<T>(4), static_cast<T>(1e-6));
    ASSERT_NEAR(stdv[4], static_cast<T>(5), static_cast<T>(1e-6));
  } else {
    ASSERT(stdv[0] == static_cast<T>(1));
    ASSERT(stdv[1] == static_cast<T>(2));
    ASSERT(stdv[2] == static_cast<T>(3));
    ASSERT(stdv[3] == static_cast<T>(4));
    ASSERT(stdv[4] == static_cast<T>(5));
  }
#endif
}

template <class T>
HOSTDEV
TEST_CASE(push_back_lval_ref)
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

#if CHECK_STD_VECTOR
  std::vector<TestStruct> v_std;
  v_std.emplace_back(1, 2.0F, 3.0);
  ASSERT(v_std.size() == 1);
  ASSERT(v_std.capacity() == 1);
  ASSERT(v_std[0].a == 1);

  std::vector<int> v2_std = {1, 2, 3};
  std::vector<std::vector<int>> v3_std;
  v3_std.emplace_back(v2_std);
  v3_std.emplace_back(std::move(v2_std));
  ASSERT(v3_std.size() == 2);
  ASSERT(v3_std.capacity() == 2);
  for (size_t i = 0; i < 2; ++i) {
    ASSERT(v3_std[i].size() == 3);
    for (size_t j = 0; j < 3; ++j) {
      ASSERT(v3_std[i][j] == static_cast<int>(j + 1));
    }
  }
#endif
}

////==============================================================================
//// CUDA
////==============================================================================
//
//#if UM2_USE_CUDA
//template <class T>
//MAKE_CUDA_KERNEL(constructor_Size, T)
//
//template <class T>
//MAKE_CUDA_KERNEL(constructor_Size_value, T)
//
//template <class T>
//MAKE_CUDA_KERNEL(copy_constructor, T)
//
//template <class T>
//MAKE_CUDA_KERNEL(move_constructor, T)
//
//template <class T>
//MAKE_CUDA_KERNEL(constructor_initializer_list, T)
//
//template <class T>
//MAKE_CUDA_KERNEL(operator_copy, T)
//
//template <class T>
//MAKE_CUDA_KERNEL(operator_move, T)
//
//template <class T>
//MAKE_CUDA_KERNEL(resize, T)
//
//template <class T>
//MAKE_CUDA_KERNEL(reserve, T)
//
//template <class T>
//MAKE_CUDA_KERNEL(push_back, T)
//
//template <class T>
//MAKE_CUDA_KERNEL(push_back_rval_ref, T)
//
//template <class T>
//MAKE_CUDA_KERNEL(push_back_n, T)
//
//MAKE_CUDA_KERNEL(clear)
//
//    MAKE_CUDA_KERNEL(emplace_back)
//#endif // UM2_USE_CUDA
//
template <class T>
TEST_SUITE(Vector)
{
  // Constructors
  TEST_HOSTDEV(constructor_n, 1, 1, T)
  TEST_HOSTDEV(constructor_n_value, 1, 1, T)
  TEST_HOSTDEV(copy_constructor, 1, 1, T)
  TEST_HOSTDEV(move_constructor, 1, 1, T)
  TEST_HOSTDEV(constructor_initializer_list, 1, 1, T)

  // Operators
  TEST_HOSTDEV(operator_copy, 1, 1, T)
  TEST_HOSTDEV(operator_move, 1, 1, T)
  TEST_HOSTDEV(operator_initializer_list, 1, 1, T)
  TEST_HOSTDEV(operator_equal)

  // Functions
  TEST_HOSTDEV(clear)
  TEST_HOSTDEV(resize, 1, 1, T)
  TEST_HOSTDEV(reserve, 1, 1, T)
  TEST_HOSTDEV(push_back, 1, 1, T)
  TEST_HOSTDEV(push_back_lval_ref, 1, 1, T)
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
