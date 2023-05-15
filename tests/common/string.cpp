#include "../test_framework.hpp"
#include <um2/common/string.hpp>

// -- Constructors --

UM2_HOSTDEV TEST_CASE(const_char_array_constructor)
{
  um2::String s("hello");
  EXPECT_EQ(s.size(), 5);
  EXPECT_EQ(s.capacity(), 6);
  EXPECT_EQ(s.data()[0], 'h');
  EXPECT_EQ(s.data()[1], 'e');
  EXPECT_EQ(s.data()[2], 'l');
  EXPECT_EQ(s.data()[3], 'l');
  EXPECT_EQ(s.data()[4], 'o');
  EXPECT_EQ(s.data()[5], '\0');
}
MAKE_CUDA_KERNEL(const_char_array_constructor);

UM2_HOSTDEV TEST_CASE(um2_string_constructor)
{
  um2::String s0("hello");
  um2::String s(s0);
  EXPECT_EQ(s.size(), 5);
  EXPECT_EQ(s.capacity(), 6);
  EXPECT_EQ(s.data()[0], 'h');
  EXPECT_EQ(s.data()[1], 'e');
  EXPECT_EQ(s.data()[2], 'l');
  EXPECT_EQ(s.data()[3], 'l');
  EXPECT_EQ(s.data()[4], 'o');
  EXPECT_EQ(s.data()[5], '\0');
}
MAKE_CUDA_KERNEL(um2_string_constructor);

// -- Operators --

UM2_HOSTDEV TEST_CASE(assign_um2_string)
{
  um2::String s0("hello");
  um2::String s = s0;
  EXPECT_EQ(s.size(), 5);
  EXPECT_EQ(s.capacity(), 6);
  EXPECT_EQ(s.data()[0], 'h');
  EXPECT_EQ(s.data()[1], 'e');
  EXPECT_EQ(s.data()[2], 'l');
  EXPECT_EQ(s.data()[3], 'l');
  EXPECT_EQ(s.data()[4], 'o');
  EXPECT_EQ(s.data()[5], '\0');
  EXPECT_NE(s.data(), s0.data());
}
MAKE_CUDA_KERNEL(assign_um2_string);

// template <typename T>
// UM2_HOSTDEV TEST(clear)
//     um2::String<T> v(10, 2);
//     v.clear();
//     ASSERT(v.size() == 0, "size is not 0");
//     ASSERT(v.capacity() == 0, "capacity is not 0");
//     ASSERT(v.data() == nullptr, "data is not nullptr");
// END_TEST
//
// template <typename T>
// UM2_HOSTDEV TEST(copy_constructor)
//     um2::String<T> v(10);
//     for (int i = 0; i < 10; i++) {
//         v.data()[i] = i;
//     }
//     um2::String<T> v2(v);
//     ASSERT(v2.size() == 10, "size is not 10");
//     ASSERT(v2.capacity() == 16, "capacity is not 16");
//     ASSERT(v2.data() != nullptr, "data is nullptr");
//     for (int i = 0; i < 10; i++) {
//         ASSERT(v2.data()[i] == i, "data[i] is not i");
//     }
//     um2::String<T> v3 = v;
//     ASSERT(v3.size() == 10, "size is not 10");
//     ASSERT(v3.capacity() == 16, "capacity is not 16");
//     ASSERT(v3.data() != nullptr, "data is nullptr");
//     for (int i = 0; i < 10; i++) {
//         ASSERT(v3.data()[i] == i, "data[i] is not i");
//     }
// END_TEST
//
// template <typename T>
// UM2_HOSTDEV TEST(initializer_list_constructor)
//     um2::String<T> v = {1, 2, 3, 4, 5};
//     ASSERT(v.size() == 5, "size is not 5");
//     ASSERT(v.capacity() == 8, "capacity is not 8");
//     ASSERT(v.data() != nullptr, "data is nullptr");
//     for (int i = 0; i < 5; i++) {
//         ASSERT(v.data()[i] == i + 1, "data[i] is not i + 1");
//     }
// END_TEST
//
// template <typename T>
// UM2_HOSTDEV TEST(reserve)
//     um2::String<T> v;
//     // Check that reserve does not change size or data, and gives the correct capacity
//     v.reserve(0);
//     ASSERT(v.size() == 0, "size");
//     ASSERT(v.capacity() == 0, "capacity");
//     ASSERT(v.data() == nullptr, "data");
//     v.reserve(1);
//     ASSERT(v.size() == 0, "size");
//     ASSERT(v.capacity() == 1, "capacity");
//     ASSERT(v.data() != nullptr, "data");
//     v.clear();
//     v.reserve(2);
//     ASSERT(v.size() == 0, "size");
//     ASSERT(v.capacity() == 2, "capacity");
//     ASSERT(v.data() != nullptr, "data");
//     v.clear();
//     v.reserve(3);
//     ASSERT(v.size() == 0, "size");
//     ASSERT(v.capacity() == 4, "capacity");
//     ASSERT(v.data() != nullptr, "data");
//     v.clear();
//     v.reserve(4);
//     ASSERT(v.size() == 0, "size");
//     ASSERT(v.capacity() == 4, "capacity");
//     ASSERT(v.data() != nullptr, "data");
//     v.clear();
//     v.reserve(15);
//     ASSERT(v.size() == 0, "size");
//     ASSERT(v.capacity() == 16, "capacity");
//     ASSERT(v.data() != nullptr, "data");
// END_TEST
//
// UM2_HOSTDEV TEST(empty)
//     um2::String<int> v;
//     ASSERT(v.empty(), "empty");
//     v.push_back(1);
//     ASSERT(!v.empty(), "!empty");
// END_TEST
//
// template <typename T>
// UM2_HOSTDEV TEST(begin_end)
//     um2::String<T> v;
//     ASSERT(v.begin() == v.end(), "begin == end");
//     v.push_back(1);
//     ASSERT(v.begin() != v.end(), "begin != end");
//     ASSERT(*v.begin() == 1, "*begin");
//     ASSERT(*(v.end() - 1) == 1, "*(end - 1)");
//     v.push_back(2);
//     ASSERT(v.begin() != v.end(), "begin != end");
//     ASSERT(*v.begin() == 1, "*begin");
//     ASSERT(*(v.end() - 1) == 2, "*(end - 1)");
//     *v.begin() = 3;
//     ASSERT(v[0] == 3, "*begin");
//
//     v.clear();
//     v.reserve(3);
//     v.push_back(1);
//     v.push_back(2);
//     v.data()[2] = 3;
//     ASSERT(v.cbegin() != v.cend(), "cbegin != cend");
//     ASSERT(*v.cbegin() == 1, "*cbegin");
//     ASSERT(*v.cend() == 3, "*(cend - 1)");
// END_TEST
//
// template <typename T>
// UM2_HOSTDEV TEST(push_back)
//     um2::String<T> v;
//     v.push_back(1);
//     ASSERT(v.size() == 1, "size");
//     ASSERT(v.capacity() == 1, "capacity");
//     ASSERT(v.data()[0] == 1, "data[0]");
//     v.push_back(2);
//     ASSERT(v.size() == 2, "size");
//     ASSERT(v.capacity() == 2, "capacity");
//     ASSERT(v.data()[0] == 1, "data[0]");
//     ASSERT(v.data()[1] == 2, "data[1]");
//     v.push_back(3);
//     ASSERT(v.size() == 3, "size");
//     ASSERT(v.capacity() == 4, "capacity");
//     ASSERT(v.data()[0] == 1, "data[0]");
//     ASSERT(v.data()[1] == 2, "data[1]");
//     ASSERT(v.data()[2] == 3, "data[2]");
//     v.clear();
//     v.reserve(3);
//     v.push_back(1);
//     v.push_back(2);
//     v.push_back(3);
//     v.push_back(4);
//     v.push_back(5);
//     ASSERT(v.size() == 5, "size");
//     ASSERT(v.capacity() == 8, "capacity");
//     ASSERT(v.data()[0] == 1, "data[0]");
//     ASSERT(v.data()[1] == 2, "data[1]");
//     ASSERT(v.data()[2] == 3, "data[2]");
//     ASSERT(v.data()[3] == 4, "data[3]");
//     ASSERT(v.data()[4] == 5, "data[4]");
// END_TEST
//
// template <typename T>
// UM2_HOSTDEV TEST(insert)
//     um2::String<T> v;
//     // Check insertion at begin
//     v.insert(v.begin(), 2, 1);
//     ASSERT(v.size() == 2, "size");
//     ASSERT(v.capacity() == 2, "capacity");
//     ASSERT(v.data()[0] == 1, "data[0]");
//     ASSERT(v.data()[1] == 1, "data[1]");
//     v.insert(v.begin(), 2, 2);
//     ASSERT(v.size() == 4, "size");
//     ASSERT(v.capacity() == 4, "capacity");
//     ASSERT(v.data()[0] == 2, "data[0]");
//     ASSERT(v.data()[1] == 2, "data[1]");
//     ASSERT(v.data()[2] == 1, "data[2]");
//     ASSERT(v.data()[3] == 1, "data[3]");
//     // Check insertion at end
//     v.clear();
//     v.insert(v.begin(), 2, 1);
//     v.insert(v.end(), 2, 2);
//     ASSERT(v.size() == 4, "size");
//     ASSERT(v.capacity() == 4, "capacity");
//     ASSERT(v.data()[0] == 1, "data[0]");
//     ASSERT(v.data()[1] == 1, "data[1]");
//     ASSERT(v.data()[2] == 2, "data[2]");
//     ASSERT(v.data()[3] == 2, "data[3]");
//     // Check insertion in middle
//     v.clear();
//     v.insert(v.begin(), 2, 1);
//     v.insert(v.begin() + 1, 2, 2);
//     ASSERT(v.size() == 4, "size");
//     ASSERT(v.capacity() == 4, "capacity");
//     ASSERT(v.data()[0] == 1, "data[0]");
//     ASSERT(v.data()[1] == 2, "data[1]");
//     ASSERT(v.data()[2] == 2, "data[2]");
//     ASSERT(v.data()[3] == 1, "data[3]");
// END_TEST
//
// UM2_HOSTDEV TEST(is_approx_int)
//     um2::String<int> v1 = {1, 2, 3, 4, 5};
//     um2::String<int> v2 = {1, 2, 3, 4, 5};
//     um2::String<int> v3 = {1, 2, 3, 4, 6};
//     um2::String<int> v4 = {1, 2, 3, 4, 5, 6};
//     ASSERT(is_approx(v1, v2), "is_approx(v1, v2)");
//     ASSERT(!is_approx(v1, v3), "!is_approx(v1, v3)");
//     ASSERT(!is_approx(v1, v4), "!is_approx(v1, v4)");
// END_TEST
//
//#if UM2_HAS_CUDA
// template <typename T>
// ADD_TEMPLATED_CUDA_KERNEL(clear, clear_kernel, T)
// template <typename T>
// ADD_TEMPLATED_KERNEL_TEST(clear_kernel, clear_cuda, T)
//
// template <typename T>
// ADD_TEMPLATED_CUDA_KERNEL(length_constructor, length_constructor_kernel, T)
// template <typename T>
// ADD_TEMPLATED_KERNEL_TEST(length_constructor_kernel, length_constructor_cuda, T)
//
// ADD_CUDA_TEST(length_val_constructor, length_val_constructor_cuda)
//
// template <typename T>
// ADD_TEMPLATED_CUDA_KERNEL(copy_constructor, copy_constructor_kernel, T)
// template <typename T>
// ADD_TEMPLATED_KERNEL_TEST(copy_constructor_kernel, copy_constructor_cuda, T)
//
// template <typename T>
// ADD_TEMPLATED_CUDA_KERNEL(initializer_list_constructor, initializer_list_constructor_kernel, T)
// template <typename T>
// ADD_TEMPLATED_KERNEL_TEST(initializer_list_constructor_kernel, initializer_list_constructor_cuda,
// T)
//
// template <typename T>
// ADD_TEMPLATED_CUDA_KERNEL(reserve, reserve_kernel, T)
// template <typename T>
// ADD_TEMPLATED_KERNEL_TEST(reserve_kernel, reserve_cuda, T)
//
// template <typename T>
// ADD_TEMPLATED_CUDA_KERNEL(push_back, push_back_kernel, T)
// template <typename T>
// ADD_TEMPLATED_KERNEL_TEST(push_back_kernel, push_back_cuda, T)
//
// template <typename T>
// ADD_TEMPLATED_CUDA_KERNEL(insert, insert_kernel, T)
// template <typename T>
// ADD_TEMPLATED_KERNEL_TEST(insert_kernel, insert_cuda, T)
//
// ADD_CUDA_TEST(empty, empty_cuda)
//
// template <typename T>
// ADD_TEMPLATED_CUDA_KERNEL(begin_end, begin_end_kernel, T)
// template <typename T>
// ADD_TEMPLATED_KERNEL_TEST(begin_end_kernel, begin_end_cuda, T)
//
// ADD_CUDA_TEST(string_char_array_assignment, string_char_array_assignment_cuda)
// ADD_CUDA_TEST(is_approx_int, is_approx_int_cuda)
//#endif
//
TEST_SUITE(string)
{
  // Constructors
  TEST_HOSTDEV(const_char_array_constructor)
  TEST_HOSTDEV(um2_string_constructor)
  // Operators
  TEST_HOSTDEV(assign_um2_string)
  //    RUN_TEST("length_constructor", length_constructor<T>);
  //    RUN_TEST("length_val_constructor", length_val_constructor);
  //    RUN_TEST("clear", clear<T>);
  //    RUN_TEST("copy_constructor", copy_constructor<T>);
  //    RUN_TEST("initializer_list_constructor", initializer_list_constructor<T>);
  //    RUN_TEST("reserve", reserve<T>);
  //    RUN_TEST("push_back", push_back<T>);
  //    RUN_TEST("empty", empty);
  //    RUN_TEST("insert", insert<T>);
  //    RUN_TEST("begin_end", begin_end<T>);
  //    RUN_TEST("is_approx_int", is_approx_int);
  //
  //    RUN_CUDA_TEST("clear_cuda", clear_cuda<T>);
  //    RUN_CUDA_TEST("length_constructor_cuda", length_constructor_cuda<T>);
  //    RUN_CUDA_TEST("length_val_constructor_cuda", length_val_constructor_cuda);
  //    RUN_CUDA_TEST("copy_constructor_cuda", copy_constructor_cuda<T>);
  //    RUN_CUDA_TEST("initializer_list_constructor_cuda", initializer_list_constructor_cuda<T>);
  //    RUN_CUDA_TEST("reserve_cuda", reserve_cuda<T>);
  //    RUN_CUDA_TEST("push_back_cuda", push_back_cuda<T>);
  //    RUN_CUDA_TEST("insert_cuda", insert_cuda<T>);
  //    RUN_CUDA_TEST("empty_cuda", empty_cuda);
  //    RUN_CUDA_TEST("begin_end_cuda", begin_end_cuda<T>);
  //    RUN_CUDA_TEST("is_approx_int_cuda", is_approx_int_cuda);
}
auto main() -> int
{
  RUN_TESTS(string)
  return 0;
}
