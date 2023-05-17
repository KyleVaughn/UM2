#include "../test_framework.hpp"
#include <um2/common/vector.hpp>

// ----------------------------------------------------------------------------
// Accessors
// ----------------------------------------------------------------------------

template <typename T>
UM2_HOSTDEV TEST_CASE(begin_end)
{
  um2::Vector<T> v;
  EXPECT_EQ(v.begin(), v.end());
  v.push_back(1);
  EXPECT_NE(v.begin(), v.end());
  EXPECT_EQ(*v.begin(), 1);
  EXPECT_EQ(*(v.end() - 1), 1);
  v.push_back(2);
  EXPECT_NE(v.begin(), v.end());
  EXPECT_EQ(*v.begin(), 1);
  EXPECT_EQ(*(v.end() - 1), 2);

  v.clear();
  EXPECT_EQ(v.cbegin(), v.cend());
  v.push_back(1);
  EXPECT_NE(v.cbegin(), v.cend());
  EXPECT_EQ(*v.cbegin(), 1);
  EXPECT_EQ(*(v.cend() - 1), 1);
  v.push_back(2);
  EXPECT_NE(v.cbegin(), v.cend());
  EXPECT_EQ(*v.cbegin(), 1);
  EXPECT_EQ(*(v.cend() - 1), 2);
}

template <typename T>
UM2_HOSTDEV TEST_CASE(front_back)
{
  um2::Vector<T> v;
  v.push_back(1);
  EXPECT_EQ(v.front(), 1);
  EXPECT_EQ(v.back(), 1);
  v.push_back(2);
  EXPECT_EQ(v.front(), 1);
  EXPECT_EQ(v.back(), 2);
}

// ----------------------------------------------------------------------------
// Constructors
// ----------------------------------------------------------------------------

template <typename T>
UM2_HOSTDEV TEST_CASE(length_constructor)
{
  um2::Vector<T> v(10);
  EXPECT_EQ(v.size(), 10);
  EXPECT_EQ(v.capacity(), 10);
  EXPECT_NE(v.data(), nullptr);
}

template <typename T>
UM2_HOSTDEV TEST_CASE(length_val_constructor)
{
  um2::Vector<T> v(10, 2);
  EXPECT_EQ(v.size(), 10);
  EXPECT_EQ(v.capacity(), 10);
  for (int i = 0; i < 10; i++) {
    EXPECT_EQ(v.data()[i], 2);
  }
}

template <typename T>
UM2_HOSTDEV TEST_CASE(copy_constructor)
{
  um2::Vector<T> v(10);
  for (int i = 0; i < 10; i++) {
    v.data()[i] = i;
  }
  um2::Vector<T> v2(v);
  EXPECT_EQ(v2.size(), 10);
  EXPECT_EQ(v2.capacity(), 16);
  for (int i = 0; i < 10; i++) {
    EXPECT_EQ(v2.data()[i], i);
  }
  um2::Vector<T> v3 = v;
  EXPECT_EQ(v3.size(), 10);
  EXPECT_EQ(v3.capacity(), 16);
  for (int i = 0; i < 10; i++) {
    EXPECT_EQ(v3.data()[i], i);
  }
}

template <typename T>
UM2_HOSTDEV TEST_CASE(initializer_list_constructor)
{
  um2::Vector<T> v{1, 2, 3, 4, 5};
  EXPECT_EQ(v.size(), 5);
  EXPECT_EQ(v.capacity(), 8);
  for (int i = 0; i < 5; i++) {
    EXPECT_EQ(v.data()[i], i + 1);
  }
}

// ----------------------------------------------------------------------------
// Operators
// ----------------------------------------------------------------------------

template <typename T>
UM2_HOSTDEV TEST_CASE(operator_equal)
{
  um2::Vector<T> v1{1, 2, 3, 4, 5};
  um2::Vector<T> v2{1, 2, 3, 4, 5};
  um2::Vector<T> v3{1, 2, 3, 4, 6};
  um2::Vector<T> v4{1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(v1 == v2);
  EXPECT_FALSE(v1 == v3);
  EXPECT_FALSE(v1 == v4);
}

template <typename T>
UM2_HOSTDEV TEST_CASE(operator_assign)
{
  um2::Vector<T> v1{1, 2, 3, 4, 5};
  um2::Vector<T> v2;
  v2 = v1;
  EXPECT_TRUE(v1 == v2);
  EXPECT_EQ(v1.size(), v2.size());
  EXPECT_EQ(v1.capacity(), v2.capacity());
  EXPECT_NE(v1.data(), v2.data());
}

// ----------------------------------------------------------------------------
// Methods
// ----------------------------------------------------------------------------

template <typename T>
UM2_HOSTDEV TEST_CASE(clear)
{
  um2::Vector<T> v(10, 2);
  v.clear();
  EXPECT_EQ(v.size(), 0);
  EXPECT_EQ(v.capacity(), 0);
  EXPECT_EQ(v.data(), nullptr);
}

template <typename T>
UM2_HOSTDEV TEST_CASE(reserve)
{
  um2::Vector<T> v;
  // Check that reserve does not change size or data, and gives the correct capacity
  v.reserve(0);
  EXPECT_EQ(v.size(), 0);
  EXPECT_EQ(v.capacity(), 0);
  EXPECT_EQ(v.data(), nullptr);
  v.reserve(1);
  EXPECT_EQ(v.size(), 0);
  EXPECT_EQ(v.capacity(), 1);
  EXPECT_NE(v.data(), nullptr);
  v.clear();
  v.reserve(2);
  EXPECT_EQ(v.size(), 0);
  EXPECT_EQ(v.capacity(), 2);
  EXPECT_NE(v.data(), nullptr);
  v.clear();
  v.reserve(3);
  EXPECT_EQ(v.size(), 0);
  EXPECT_EQ(v.capacity(), 4);
  EXPECT_NE(v.data(), nullptr);
  v.clear();
  v.reserve(4);
  EXPECT_EQ(v.size(), 0);
  EXPECT_EQ(v.capacity(), 4);
  EXPECT_NE(v.data(), nullptr);
  v.clear();
  v.reserve(15);
  EXPECT_EQ(v.size(), 0);
  EXPECT_EQ(v.capacity(), 16);
  EXPECT_NE(v.data(), nullptr);
}

template <typename T>
UM2_HOSTDEV TEST_CASE(resize)
{
  um2::Vector<T> v;
  v.resize(0);
  EXPECT_EQ(v.size(), 0);
  EXPECT_EQ(v.capacity(), 0);
  EXPECT_EQ(v.data(), nullptr);
  v.resize(1);
  EXPECT_EQ(v.size(), 1);
  EXPECT_EQ(v.capacity(), 1);
  EXPECT_NE(v.data(), nullptr);
  v.resize(2);
  EXPECT_EQ(v.size(), 2);
  EXPECT_EQ(v.capacity(), 2);
  EXPECT_NE(v.data(), nullptr);
  v.resize(3);
  EXPECT_EQ(v.size(), 3);
  EXPECT_EQ(v.capacity(), 4);
  EXPECT_NE(v.data(), nullptr);
}

template <typename T>
UM2_HOSTDEV TEST_CASE(push_back)
{
  um2::Vector<T> v;
  v.push_back(1);
  EXPECT_EQ(v.size(), 1);
  EXPECT_EQ(v.capacity(), 1);
  EXPECT_EQ(v.data()[0], 1);
  v.push_back(2);
  EXPECT_EQ(v.size(), 2);
  EXPECT_EQ(v.capacity(), 2);
  EXPECT_EQ(v.data()[0], 1);
  EXPECT_EQ(v.data()[1], 2);
  v.push_back(3);
  EXPECT_EQ(v.size(), 3);
  EXPECT_EQ(v.capacity(), 4);
  EXPECT_EQ(v.data()[0], 1);
  EXPECT_EQ(v.data()[1], 2);
  EXPECT_EQ(v.data()[2], 3);
  v.clear();
  v.reserve(3);
  v.push_back(1);
  v.push_back(2);
  v.push_back(3);
  v.push_back(4);
  v.push_back(5);
  EXPECT_EQ(v.size(), 5);
  EXPECT_EQ(v.capacity(), 8);
  EXPECT_EQ(v.data()[0], 1);
  EXPECT_EQ(v.data()[1], 2);
  EXPECT_EQ(v.data()[2], 3);
  EXPECT_EQ(v.data()[3], 4);
  EXPECT_EQ(v.data()[4], 5);
}

template <typename T>
UM2_HOSTDEV TEST_CASE(empty)
{
  um2::Vector<T> v;
  EXPECT_TRUE(v.empty());
  v.push_back(1);
  EXPECT_FALSE(v.empty());
}

template <typename T>
UM2_HOSTDEV TEST_CASE(insert)
{
  um2::Vector<T> v;
  // Check insertion at begin
  v.insert(v.begin(), 2, 1);
  EXPECT_EQ(v.size(), 2);
  EXPECT_EQ(v.capacity(), 2);
  EXPECT_EQ(v.data()[0], 1);
  EXPECT_EQ(v.data()[1], 1);
  v.insert(v.begin(), 2, 2);
  EXPECT_EQ(v.size(), 4);
  EXPECT_EQ(v.capacity(), 4);
  EXPECT_EQ(v.data()[0], 2);
  EXPECT_EQ(v.data()[1], 2);
  EXPECT_EQ(v.data()[2], 1);
  EXPECT_EQ(v.data()[3], 1);
  // Check insertion at end
  v.clear();
  v.insert(v.begin(), 2, 1);
  v.insert(v.end(), 2, 2);
  EXPECT_EQ(v.size(), 4);
  EXPECT_EQ(v.capacity(), 4);
  EXPECT_EQ(v.data()[0], 1);
  EXPECT_EQ(v.data()[1], 1);
  EXPECT_EQ(v.data()[2], 2);
  EXPECT_EQ(v.data()[3], 2);
  // Check insertion in middle
  v.clear();
  v.insert(v.begin(), 2, 1);
  v.insert(v.begin() + 1, 2, 2);
  EXPECT_EQ(v.size(), 4);
  EXPECT_EQ(v.capacity(), 4);
  EXPECT_EQ(v.data()[0], 1);
  EXPECT_EQ(v.data()[1], 2);
  EXPECT_EQ(v.data()[2], 2);
  EXPECT_EQ(v.data()[3], 1);
}

template <typename T>
UM2_HOSTDEV TEST_CASE(contains)
{
  um2::Vector<T> v{2, 1, 5, 6};
  EXPECT_TRUE(v.contains(2));
  EXPECT_TRUE(v.contains(1));
  EXPECT_TRUE(v.contains(5));
  EXPECT_TRUE(v.contains(6));
  EXPECT_FALSE(v.contains(3));
}

template <typename T>
UM2_HOSTDEV TEST_CASE(is_approx)
{
  um2::Vector<T> v1{1, 2, 3, 4, 5};
  um2::Vector<T> v2{1, 2, 3, 4, 5};
  um2::Vector<T> v3{1, 2, 3, 4, 6};
  um2::Vector<T> v4{1, 2, 3, 4, 5, 6};
  EXPECT_TRUE(is_approx(v1, v2));
  EXPECT_FALSE(is_approx(v1, v3));
  EXPECT_FALSE(is_approx(v1, v4));
  EXPECT_TRUE(is_approx(v1, v2, 1));
  EXPECT_TRUE(is_approx(v1, v3, 1));
}

// --------------------------------------------------------------------------
// CUDA
// --------------------------------------------------------------------------
#if UM2_ENABLE_CUDA
template <typename T>
MAKE_CUDA_KERNEL(begin_end, T)

template <typename T>
MAKE_CUDA_KERNEL(front_back, T)

template <typename T>
MAKE_CUDA_KERNEL(length_constructor, T)

template <typename T>
MAKE_CUDA_KERNEL(length_val_constructor, T)

template <typename T>
MAKE_CUDA_KERNEL(copy_constructor, T)

template <typename T>
MAKE_CUDA_KERNEL(initializer_list_constructor, T)

template <typename T>
MAKE_CUDA_KERNEL(clear, T)

template <typename T>
MAKE_CUDA_KERNEL(reserve, T)

template <typename T>
MAKE_CUDA_KERNEL(resize, T)

template <typename T>
MAKE_CUDA_KERNEL(push_back, T)

template <typename T>
MAKE_CUDA_KERNEL(empty, T)

template <typename T>
MAKE_CUDA_KERNEL(insert, T)

template <typename T>
MAKE_CUDA_KERNEL(contains, T)

template <typename T>
MAKE_CUDA_KERNEL(is_approx, T)

template <typename T>
MAKE_CUDA_KERNEL(operator_equal, T)

template <typename T>
MAKE_CUDA_KERNEL(operator_assign, T)

#endif // UM2_ENABLE_CUDA

template <typename T>
TEST_SUITE(vector)
{
  // Accessors
  TEST_HOSTDEV(begin_end, 1, 1, T)
  TEST_HOSTDEV(front_back, 1, 1, T)
  // Constructors
  TEST_HOSTDEV(length_constructor, 1, 1, T)
  TEST_HOSTDEV(length_val_constructor, 1, 1, T)
  TEST_HOSTDEV(copy_constructor, 1, 1, T)
  TEST_HOSTDEV(initializer_list_constructor, 1, 1, T)
  // Operators
  TEST_HOSTDEV(operator_equal, 1, 1, T)
  TEST_HOSTDEV(operator_assign, 1, 1, T)
  // Methods
  TEST_HOSTDEV(clear, 1, 1, T)
  TEST_HOSTDEV(reserve, 1, 1, T)
  TEST_HOSTDEV(resize, 1, 1, T)
  TEST_HOSTDEV(push_back, 1, 1, T)
  TEST_HOSTDEV(empty, 1, 1, T)
  TEST_HOSTDEV(insert, 1, 1, T)
  TEST_HOSTDEV(contains, 1, 1, T)
  TEST_HOSTDEV(is_approx, 1, 1, T)
}

auto main() -> int
{
  RUN_TESTS(vector<int>);
  return 0;
}
