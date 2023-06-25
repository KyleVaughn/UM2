#include "../test_macros.hpp"
#include <um2/common/Vector.hpp>

#include <um2/common/utility.hpp>

#include <concepts>

// ----------------------------------------------------------------------------
// Constructors
// ----------------------------------------------------------------------------

template <class T>
HOSTDEV
TEST_CASE(test_constructor_Size)
{
  um2::Vector<T> v(10);
  assert(v.cbegin() != nullptr);
  assert(v.cend() != nullptr);
  assert(v.size() == 10);
  assert(v.capacity() == 10);
}

template <class T>
HOSTDEV TEST_CASE(test_constructor_Size_value)
{
  um2::Vector<T> v(10, 2);
  assert(v.cbegin() != nullptr);
  assert(v.cend() != nullptr);
  assert(v.size() == 10);
  assert(v.capacity() == 10);
  for (int i = 0; i < 10; ++i) {
    if constexpr (std::floating_point<T>) {
      // cppcheck-suppress assertWithSideEffect
      EXPECT_NEAR(v.data()[i], static_cast<T>(2), static_cast<T>(1e-6));
    } else {
      // cppcheck-suppress assertWithSideEffect
      assert(v.data()[i] == 2);
    }
  }
}

template <class T>
HOSTDEV TEST_CASE(test_copy_constructor)
{
  um2::Vector<T> v(10);
  for (int i = 0; i < 10; i++) {
    v.data()[i] = static_cast<T>(i);
  }
  um2::Vector<T> v2(v);
  assert(v2.size() == 10);
  assert(v2.capacity() == 10);
  // cppcheck-suppress mismatchingContainerExpression
  assert(v.cbegin() != v2.cbegin());
  for (int i = 0; i < 10; i++) {
    if constexpr (std::floating_point<T>) {
      // cppcheck-suppress assertWithSideEffect
      EXPECT_NEAR(v2.data()[i], static_cast<T>(i), static_cast<T>(1e-6)); 
    } else {
      // cppcheck-suppress assertWithSideEffect
      assert(v2.data()[i] == static_cast<T>(i));
    }
  }

  um2::Vector<T> v3(33);
  v3 = v;
  assert(v3.size() == 10);
  assert(v3.capacity() == 10);
  // cppcheck-suppress mismatchingContainerExpression
  assert(v.cbegin() != v3.cbegin());
  // cppcheck-suppress assertWithSideEffect
  assert(&v3.front() != &v2.front());
  for (int i = 0; i < 10; i++) {
    if constexpr (std::floating_point<T>) {
      // cppcheck-suppress assertWithSideEffect
      EXPECT_NEAR(v3.data()[i], static_cast<T>(i), static_cast<T>(1e-6)); 
    } else {
      // cppcheck-suppress assertWithSideEffect
      assert(v3.data()[i] == static_cast<T>(i));
    }
  }
}

template <class T>
HOSTDEV auto createVector(Size size) -> um2::Vector<T> 
{
  um2::Vector<T> v(size);
  for (Size i = 0; i < size; i++) {
    v.data()[i] = static_cast<T>(i);
  }
  return v;
}

template <class T>
HOSTDEV TEST_CASE(test_move_constructor)
{
  um2::Vector<T> v(move(createVector<T>(10)));
  assert(v.cbegin() != nullptr);
  assert(v.cend() != nullptr);
  assert(v.size() == 10);
  assert(v.capacity() == 10);

  um2::Vector<T> v1;
  v1 = move(createVector<T>(10));
  assert(v1.cbegin() != nullptr);
  assert(v1.cend() != nullptr);
  assert(v1.size() == 10);
  assert(v1.capacity() == 10);
}

template <class T>
HOSTDEV
TEST_CASE(test_constructor_initializer_list)
{
  um2::Vector<T> v{1, 2, 3, 4, 5};
  assert(v.size() == 5);
  assert(v.capacity() == 5);
  for (int i = 0; i < 5; ++i) {
    if constexpr (std::floating_point<T>) {
      // cppcheck-suppress assertWithSideEffect
      EXPECT_NEAR(v.data()[i], static_cast<T>(i + 1), static_cast<T>(1e-6));
    } else {
      // cppcheck-suppress assertWithSideEffect
      assert(v.data()[i] == static_cast<T>(i + 1));
    }
  }
}

//// ----------------------------------------------------------------------------
//// Operators
//// ----------------------------------------------------------------------------
//
// template <class T>
// HOSTDEV TEST_CASE(operator_equal)
//{
//  um2::Vector<T> v1(5); // {1, 2, 3, 4, 5};
//  um2::Vector<T> v2(5); // {1, 2, 3, 4, 5};
//  um2::Vector<T> v3(5); // {1, 2, 3, 4, 6};
//  um2::Vector<T> v4(5); // {1, 2, 3, 4, 5, 6};
//  for (int i = 0; i < 5; i++) {
//    v1.data()[i] = static_cast<T>(i) + 1;
//    v2.data()[i] = static_cast<T>(i) + 1;
//    v3.data()[i] = static_cast<T>(i) + 1;
//    v4.data()[i] = static_cast<T>(i) + 1;
//  }
//  v3.data()[4] = 6;
//  v4.push_back(6);
//  EXPECT_TRUE(v1 == v2);
//  EXPECT_FALSE(v1 == v3);
//  EXPECT_FALSE(v1 == v4);
//}
//
// template <class T>
// HOSTDEV TEST_CASE(operator_assign)
//{
//  um2::Vector<T> v1(5);
//  for (int i = 0; i < 5; i++) {
//    v1.data()[i] = static_cast<T>(i) + 1;
//  }
//  um2::Vector<T> v2;
//  v2 = v1;
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(v2.data()[0], 1, 1e-6);
//    EXPECT_NEAR(v2.data()[1], 2, 1e-6);
//    EXPECT_NEAR(v2.data()[2], 3, 1e-6);
//    EXPECT_NEAR(v2.data()[3], 4, 1e-6);
//    EXPECT_NEAR(v2.data()[4], 5, 1e-6);
//  } else {
//    assert(v2.data()[0], 1);
//    assert(v2.data()[1], 2);
//    assert(v2.data()[2], 3);
//    assert(v2.data()[3], 4);
//    assert(v2.data()[4], 5);
//  }
//  assert(v1.size(), v2.size());
//  assert(v1.capacity(), 8);
//  EXPECT_NE(v1.data(), v2.data());
//}
//
// ----------------------------------------------------------------------------
// Methods
// ----------------------------------------------------------------------------

// NOLINTBEGIN
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

HOSTDEV TEST_CASE(test_clear)
{
  count = 0;
  um2::Vector<Counted> v(10);
  assert(count == 10);
  v.clear();
  assert(count == 0);
  assert(v.empty());
  assert(v.capacity() == 10);
}
//
// template <class T>
// HOSTDEV TEST_CASE(reserve)
//{
//  um2::Vector<T> v;
//  // Check that reserve does not change size or data, and gives the correct capacity
//  v.reserve(0);
//  assert(v.size(), 0);
//  assert(v.capacity(), 0);
//  assert(v.data(), nullptr);
//  v.reserve(1);
//  assert(v.size(), 0);
//  assert(v.capacity(), 1);
//  EXPECT_NE(v.data(), nullptr);
//  v.clear();
//  v.reserve(2);
//  assert(v.size(), 0);
//  assert(v.capacity(), 2);
//  EXPECT_NE(v.data(), nullptr);
//  v.clear();
//  v.reserve(3);
//  assert(v.size(), 0);
//  assert(v.capacity(), 4);
//  EXPECT_NE(v.data(), nullptr);
//  v.clear();
//  v.reserve(4);
//  assert(v.size(), 0);
//  assert(v.capacity(), 4);
//  EXPECT_NE(v.data(), nullptr);
//  v.clear();
//  v.reserve(15);
//  assert(v.size(), 0);
//  assert(v.capacity(), 16);
//  EXPECT_NE(v.data(), nullptr);
//}
//
// template <class T>
// HOSTDEV TEST_CASE(resize)
//{
//  um2::Vector<T> v;
//  v.resize(0);
//  assert(v.size(), 0);
//  assert(v.capacity(), 0);
//  assert(v.data(), nullptr);
//  v.resize(1);
//  assert(v.size(), 1);
//  assert(v.capacity(), 1);
//  EXPECT_NE(v.data(), nullptr);
//  v.resize(2);
//  assert(v.size(), 2);
//  assert(v.capacity(), 2);
//  EXPECT_NE(v.data(), nullptr);
//  v.resize(3);
//  assert(v.size(), 3);
//  assert(v.capacity(), 4);
//  EXPECT_NE(v.data(), nullptr);
//}
//
// template <class T>
// HOSTDEV TEST_CASE(push_back)
//{
//  um2::Vector<T> v;
//  v.push_back(1);
//  assert(v.size(), 1);
//  assert(v.capacity(), 1);
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(v.data()[0], static_cast<T>(1), 1e-6);
//  } else {
//    assert(v.data()[0], static_cast<T>(1));
//  }
//  v.push_back(2);
//  assert(v.size(), 2);
//  assert(v.capacity(), 2);
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(v.data()[0], static_cast<T>(1), 1e-6);
//    EXPECT_NEAR(v.data()[1], static_cast<T>(2), 1e-6);
//  } else {
//    assert(v.data()[0], static_cast<T>(1));
//    assert(v.data()[1], static_cast<T>(2));
//  }
//  v.push_back(3);
//  assert(v.size(), 3);
//  assert(v.capacity(), 4);
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(v.data()[0], static_cast<T>(1), 1e-6);
//    EXPECT_NEAR(v.data()[1], static_cast<T>(2), 1e-6);
//    EXPECT_NEAR(v.data()[2], static_cast<T>(3), 1e-6);
//  } else {
//    assert(v.data()[0], static_cast<T>(1));
//    assert(v.data()[1], static_cast<T>(2));
//    assert(v.data()[2], static_cast<T>(3));
//  }
//  v.clear();
//  v.reserve(3);
//  v.push_back(1);
//  v.push_back(2);
//  v.push_back(3);
//  v.push_back(4);
//  v.push_back(5);
//  assert(v.size(), 5);
//  assert(v.capacity(), 8);
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(v.data()[0], static_cast<T>(1), 1e-6);
//    EXPECT_NEAR(v.data()[1], static_cast<T>(2), 1e-6);
//    EXPECT_NEAR(v.data()[2], static_cast<T>(3), 1e-6);
//    EXPECT_NEAR(v.data()[3], static_cast<T>(4), 1e-6);
//    EXPECT_NEAR(v.data()[4], static_cast<T>(5), 1e-6);
//  } else {
//    assert(v.data()[0], static_cast<T>(1));
//    assert(v.data()[1], static_cast<T>(2));
//    assert(v.data()[2], static_cast<T>(3));
//    assert(v.data()[3], static_cast<T>(4));
//    assert(v.data()[4], static_cast<T>(5));
//  }
//}
//
// template <class T>
// HOSTDEV TEST_CASE(insert)
//{
//  um2::Vector<T> v;
//  // Check insertion at begin
//  v.insert(v.begin(), 2, 1);
//  assert(v.size(), 2);
//  assert(v.capacity(), 2);
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(v.data()[0], static_cast<T>(1), 1e-6);
//    EXPECT_NEAR(v.data()[1], static_cast<T>(1), 1e-6);
//  } else {
//    assert(v.data()[0], static_cast<T>(1));
//    assert(v.data()[1], static_cast<T>(1));
//  }
//  v.insert(v.begin(), 2, 2);
//  assert(v.size(), 4);
//  assert(v.capacity(), 4);
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(v.data()[0], static_cast<T>(2), 1e-6);
//    EXPECT_NEAR(v.data()[1], static_cast<T>(2), 1e-6);
//    EXPECT_NEAR(v.data()[2], static_cast<T>(1), 1e-6);
//    EXPECT_NEAR(v.data()[3], static_cast<T>(1), 1e-6);
//  } else {
//    assert(v.data()[0], static_cast<T>(2));
//    assert(v.data()[1], static_cast<T>(2));
//    assert(v.data()[2], static_cast<T>(1));
//    assert(v.data()[3], static_cast<T>(1));
//  }
//  // Check insertion at end
//  v.clear();
//  v.insert(v.begin(), 2, 1);
//  v.insert(v.end(), 2, 2);
//  assert(v.size(), 4);
//  assert(v.capacity(), 4);
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(v.data()[0], static_cast<T>(1), 1e-6);
//    EXPECT_NEAR(v.data()[1], static_cast<T>(1), 1e-6);
//    EXPECT_NEAR(v.data()[2], static_cast<T>(2), 1e-6);
//    EXPECT_NEAR(v.data()[3], static_cast<T>(2), 1e-6);
//  } else {
//    assert(v.data()[0], static_cast<T>(1));
//    assert(v.data()[1], static_cast<T>(1));
//    assert(v.data()[2], static_cast<T>(2));
//    assert(v.data()[3], static_cast<T>(2));
//  }
//  // Check insertion in middle
//  v.clear();
//  v.insert(v.begin(), 2, 1);
//  v.insert(v.begin() + 1, 2, 2);
//  assert(v.size(), 4);
//  assert(v.capacity(), 4);
//  if constexpr (std::floating_point<T>) {
//    EXPECT_NEAR(v.data()[0], static_cast<T>(1), 1e-6);
//    EXPECT_NEAR(v.data()[1], static_cast<T>(2), 1e-6);
//    EXPECT_NEAR(v.data()[2], static_cast<T>(2), 1e-6);
//    EXPECT_NEAR(v.data()[3], static_cast<T>(1), 1e-6);
//  } else {
//    assert(v.data()[0], static_cast<T>(1));
//    assert(v.data()[1], static_cast<T>(2));
//    assert(v.data()[2], static_cast<T>(2));
//    assert(v.data()[3], static_cast<T>(1));
//  }
//}
//
// template <class T>
// HOSTDEV TEST_CASE(contains)
//{
//  um2::Vector<T> v(4); // {2, 1, 5, 6};
//  v[0] = 2;
//  v[1] = 1;
//  v[2] = 5;
//  v[3] = 6;
//  EXPECT_TRUE(v.contains(2));
//  EXPECT_TRUE(v.contains(1));
//  EXPECT_TRUE(v.contains(5));
//  EXPECT_TRUE(v.contains(6));
//  EXPECT_FALSE(v.contains(3));
//}
//
// template <class T>
// HOSTDEV TEST_CASE(isApprox)
//{
//  um2::Vector<T> v1(5); // {1, 2, 3, 4, 5};
//  um2::Vector<T> v2(5); // {1, 2, 3, 4, 5};
//  um2::Vector<T> v3(5); // {1, 2, 3, 4, 6};
//  um2::Vector<T> v4(5); // {1, 2, 3, 4, 5, 6};
//  for (int i = 0; i < 5; i++) {
//    v1.data()[i] = static_cast<T>(i) + 1;
//    v2.data()[i] = static_cast<T>(i) + 1;
//    v3.data()[i] = static_cast<T>(i) + 1;
//    v4.data()[i] = static_cast<T>(i) + 1;
//  }
//  v3.data()[4] = 6;
//  v4.push_back(6);
//
//  EXPECT_TRUE(isApprox(v1, v2));
//  EXPECT_FALSE(isApprox(v1, v3));
//  EXPECT_FALSE(isApprox(v1, v4));
//  T const eps = static_cast<T>(1);
//  EXPECT_TRUE(isApprox(v1, v2, eps));
//  EXPECT_TRUE(isApprox(v1, v3, eps));
//}

// --------------------------------------------------------------------------
// CUDA
// --------------------------------------------------------------------------
#if UM2_ENABLE_CUDA
template <class T>
MAKE_CUDA_KERNEL(test_constructor_Size, T)

template <class T>
MAKE_CUDA_KERNEL(test_constructor_Size_value, T)

template <class T>
MAKE_CUDA_KERNEL(test_copy_constructor, T)

template <class T>
MAKE_CUDA_KERNEL(test_move_constructor, T)

template <class T>
MAKE_CUDA_KERNEL(test_constructor_initializer_list, T)

MAKE_CUDA_KERNEL(test_clear)
//
//  template <class T>
//  MAKE_CUDA_KERNEL(reserve, T)
//
//  template <class T>
//  MAKE_CUDA_KERNEL(resize, T)
//
//  template <class T>
//  MAKE_CUDA_KERNEL(push_back, T)
//
//  template <class T>
//  MAKE_CUDA_KERNEL(empty, T)
//
//  template <class T>
//  MAKE_CUDA_KERNEL(insert, T)
//
//  template <class T>
//  MAKE_CUDA_KERNEL(contains, T)
//
//  template <class T>
//  MAKE_CUDA_KERNEL(isApprox, T)
//
//  template <class T>
//  MAKE_CUDA_KERNEL(operator_equal, T)
//
//  template <class T>
//  MAKE_CUDA_KERNEL(operator_assign, T)

#endif // UM2_ENABLE_CUDA

template <class T>
TEST_SUITE(vector)
{
  // Constructors
  TEST_HOSTDEV(test_constructor_Size, 1, 1, T)
  TEST_HOSTDEV(test_constructor_Size_value, 1, 1, T)
  TEST_HOSTDEV(test_copy_constructor, 1, 1, T)
  TEST_HOSTDEV(test_move_constructor, 1, 1, T)
  TEST_HOSTDEV(test_constructor_initializer_list, 1, 1, T)

  //  // Operators
  //  if constexpr (!std::floating_point<T>) {
  //    TEST_HOSTDEV(operator_equal, 1, 1, T)
  //  }
  //  TEST_HOSTDEV(operator_assign, 1, 1, T)

  // Methods
  TEST_HOSTDEV(test_clear)
  //  TEST_HOSTDEV(reserve, 1, 1, T)
  //  TEST_HOSTDEV(resize, 1, 1, T)
  //  TEST_HOSTDEV(push_back, 1, 1, T)
  //  TEST_HOSTDEV(empty, 1, 1, T)
  //  TEST_HOSTDEV(insert, 1, 1, T)
  //  if constexpr (!std::floating_point<T>) {
  //    TEST_HOSTDEV(contains, 1, 1, T)
  //  }
  //  TEST_HOSTDEV(isApprox, 1, 1, T)
}

auto
main() -> int
{
  RUN_TESTS(vector<int32_t>);
  //  RUN_TESTS(vector<uint32_t>);
  //  RUN_TESTS(vector<int64_t>);
  //  RUN_TESTS(vector<uint64_t>);
  RUN_TESTS(vector<float>);
  //  RUN_TESTS(vector<double>);
  return 0;
}
