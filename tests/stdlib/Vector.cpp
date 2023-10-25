#include <um2/stdlib/Vector.hpp>
#include <um2/stdlib/utility.hpp>

#include <concepts>
#include <vector>

#include "../test_macros.hpp"

//==============================================================================
// Constructors
//==============================================================================

template <class T>
HOSTDEV
TEST_CASE(constructor_Size)
{
  um2::Vector<T> const v(10);
  ASSERT(v.cbegin() != nullptr);
  ASSERT(v.cend() != nullptr);
  ASSERT(v.size() == 10);
  ASSERT(v.capacity() == 10);
}

template <class T>
HOSTDEV
TEST_CASE(constructor_Size_value)
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
}

template <class T>
HOSTDEV
TEST_CASE(copy_constructor)
{
  um2::Vector<T> v(10);
  for (int i = 0; i < 10; i++) {
    v.data()[i] = static_cast<T>(i);
  }
  um2::Vector<T> v2(v);
  ASSERT(v2.size() == 10);
  ASSERT(v2.capacity() == 10);
  // cppcheck-suppress mismatchingContainerExpression; justification: false positive
  ASSERT(v.cbegin() != v2.cbegin());
  for (int i = 0; i < 10; i++) {
    if constexpr (std::floating_point<T>) {
      ASSERT_NEAR(v2[i], static_cast<T>(i), static_cast<T>(1e-6));
    } else {
      ASSERT(v2[i] == static_cast<T>(i));
    }
  }
}

template <class T>
HOSTDEV auto
createVector(Size size) -> um2::Vector<T>
{
  um2::Vector<T> v(size);
  for (Size i = 0; i < size; i++) {
    v[i] = static_cast<T>(i);
  }
  return v;
}

template <class T>
HOSTDEV
TEST_CASE(move_constructor)
{
  um2::Vector<T> const v(move(createVector<T>(10)));
  ASSERT(v.cbegin() != nullptr);
  ASSERT(v.cend() != nullptr);
  ASSERT(v.size() == 10);
  ASSERT(v.capacity() == 10);

  um2::Vector<T> v1;
  v1 = move(createVector<T>(10));
  ASSERT(v1.cbegin() != nullptr);
  ASSERT(v1.cend() != nullptr);
  ASSERT(v1.size() == 10);
  ASSERT(v1.capacity() == 10);
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
    v.data()[i] = static_cast<T>(i);
  }

  um2::Vector<T> v3(33);
  v3 = v;
  ASSERT(v3.size() == 10);
  ASSERT(v3.capacity() == 10);
  // cppcheck-suppress mismatchingContainerExpression; justification: false positive
  ASSERT(v.cbegin() != v3.cbegin());
  for (int i = 0; i < 10; i++) {
    if constexpr (std::floating_point<T>) {
      ASSERT_NEAR(v3[i], static_cast<T>(i), static_cast<T>(1e-6));
    } else {
      ASSERT(v3[i] == static_cast<T>(i));
    }
  }
}

template <class T>
HOSTDEV
TEST_CASE(operator_move)
{
  um2::Vector<T> v1;
  v1 = move(createVector<T>(10));
  ASSERT(v1.cbegin() != nullptr);
  ASSERT(v1.cend() != nullptr);
  ASSERT(v1.size() == 10);
  ASSERT(v1.capacity() == 10);
}

//==============================================================================--
// Methods
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
  my_vector.push_back(std::move(l_value_vector));
  ASSERT(my_vector.size() == 1);
  ASSERT(my_vector.capacity() == 1);
  if constexpr (std::floating_point<T>) {
    ASSERT_NEAR(my_vector.data()[0].data()[0], static_cast<T>(1), static_cast<T>(1e-6));
  } else {
    ASSERT(my_vector.data()[0].data()[0] == static_cast<T>(1));
  }
}

template <class T>
HOSTDEV
TEST_CASE(push_back_n)
{
  um2::Vector<T> empty_vector;
  um2::Vector<T> non_empty_vector{1, 2, 3};
  empty_vector.push_back(3, static_cast<T>(7));
  ASSERT(empty_vector.size() == 3);
  for (const auto & i : empty_vector) {
    if constexpr (std::floating_point<T>) {
      ASSERT_NEAR(i, static_cast<T>(7), static_cast<T>(1e-6));
    } else {
      ASSERT(i == static_cast<T>(7));
    }
  }
  non_empty_vector.push_back(2, static_cast<T>(5));
  ASSERT(non_empty_vector.size() == 5);
  ASSERT(non_empty_vector.capacity() == 6);
}

HOSTDEV
TEST_CASE(emplace_back)
{
  struct TestStruct {
    int a;
    float b;
    double c;
    TestStruct(int ia, float ib, double ic) : a(ia), b(ib), c(ic) {}
  };
  um2::Vector<TestStruct> v;
  v.emplace_back(1, 2.0F, 3.0);
  ASSERT(v.size() == 1);
  ASSERT(v.capacity() == 1);
  ASSERT(v.data()[0].a == 1);
}

//==============================================================================
// CUDA
//==============================================================================
#if UM2_USE_CUDA
template <class T>
MAKE_CUDA_KERNEL(constructor_Size, T)

template <class T>
MAKE_CUDA_KERNEL(constructor_Size_value, T)

template <class T>
MAKE_CUDA_KERNEL(copy_constructor, T)

template <class T>
MAKE_CUDA_KERNEL(move_constructor, T)

template <class T>
MAKE_CUDA_KERNEL(constructor_initializer_list, T)

template <class T>
MAKE_CUDA_KERNEL(operator_copy, T)

template <class T>
MAKE_CUDA_KERNEL(operator_move, T)

MAKE_CUDA_KERNEL(clear)

    template <class T>
    MAKE_CUDA_KERNEL(resize, T)

    template <class T>
    MAKE_CUDA_KERNEL(push_back, T)

    template <class T>
    MAKE_CUDA_KERNEL(push_back_rval_ref, T)

    template <class T>
    MAKE_CUDA_KERNEL(push_back_n, T)

MAKE_CUDA_KERNEL(emplace_back)
#endif // UM2_USE_CUDA

    template <class T>
    TEST_SUITE(Vector)
{
  // Constructors
  TEST_HOSTDEV(constructor_Size, 1, 1, T)
  TEST_HOSTDEV(constructor_Size_value, 1, 1, T)
  TEST_HOSTDEV(copy_constructor, 1, 1, T)
  TEST_HOSTDEV(move_constructor, 1, 1, T)
  TEST_HOSTDEV(constructor_initializer_list, 1, 1, T)

  // Operators
  TEST_HOSTDEV(operator_copy, 1, 1, T)
  TEST_HOSTDEV(operator_move, 1, 1, T)

  // Methods
  TEST_HOSTDEV(clear)
  TEST_HOSTDEV(resize, 1, 1, T)
  TEST_HOSTDEV(push_back, 1, 1, T)
  TEST_HOSTDEV(push_back_rval_ref, 1, 1, T)
  TEST_HOSTDEV(push_back_n, 1, 1, T)
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
