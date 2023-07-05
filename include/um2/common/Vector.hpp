#pragma once

#include <um2/common/algorithm.hpp> // copy
#include <um2/common/memory.hpp>    // addressof
#include <um2/common/utility.hpp>   // move

#include <thrust/extrema.h> // thrust::max

#include <cuda/std/bit>     // cuda::std::bit_ceil
#include <cuda/std/utility> // cuda::std::pair

#include <initializer_list> // std::initializer_list

namespace um2
{

// -----------------------------------------------------------------------------
// VECTOR
// -----------------------------------------------------------------------------
// An std::vector-like class without and Allocator template parameter.
//
// https://en.cppreference.com/w/cpp/container/vector

template <typename T>
struct Vector {

  using Ptr = T *;
  using ConstPtr = T const *;

private:
  Ptr _begin = nullptr;
  Ptr _end = nullptr;
  Ptr _end_cap = nullptr;

public:
  // -----------------------------------------------------------------------------
  // Constructors
  // -----------------------------------------------------------------------------

  constexpr Vector() noexcept = default;

  HOSTDEV explicit constexpr Vector(Size n) noexcept;

  HOSTDEV constexpr Vector(Size n, T const & value) noexcept;

  HOSTDEV constexpr Vector(Vector const & v) noexcept;

  HOSTDEV constexpr Vector(Vector && v) noexcept;

  // cppcheck-suppress noExplicitConstructor
  HOSTDEV constexpr Vector(std::initializer_list<T> const & list) noexcept;

  // -----------------------------------------------------------------------------
  // Destructor
  // -----------------------------------------------------------------------------

  HOSTDEV constexpr ~Vector() noexcept;

  // -----------------------------------------------------------------------------
  // Accessors
  // -----------------------------------------------------------------------------

  PURE HOSTDEV [[nodiscard]] static constexpr auto
  // NOLINTNEXTLINE(readability-identifier-naming)
  max_size() noexcept -> Size;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  // cppcheck-suppress functionConst
  begin() noexcept -> T *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  begin() const noexcept -> T const *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  // cppcheck-suppress functionConst
  end() noexcept -> T *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  end() const noexcept -> T const *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  size() const noexcept -> Size;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  capacity() const noexcept -> Size;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  empty() const noexcept -> bool;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  cbegin() const noexcept -> T const *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  cend() const noexcept -> T const *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  // cppcheck-suppress functionConst
  front() noexcept -> T &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  front() const noexcept -> T const &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  // cppcheck-suppress functionConst
  back() noexcept -> T &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  back() const noexcept -> T const &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  // cppcheck-suppress functionConst
  data() noexcept -> T *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  data() const noexcept -> T const *;

  // -----------------------------------------------------------------------------
  // Methods
  // -----------------------------------------------------------------------------

  //    void reserve(size_type n);
  //    void shrink_to_fit() noexcept;
  //    void push_back(const value_type& x);
  //    void push_back(value_type&& x);
  //    template <class... Args>
  //        reference emplace_back(Args&&... args); // reference in C++17
  //    template<container-compatible-range<T> R>
  //      constexpr void append_range(R&& rg); // C++23
  //    void pop_back();
  //    iterator insert(const_iterator position, const value_type& x);
  //    iterator insert(const_iterator position, value_type&& x);
  //    iterator insert(const_iterator position, size_type n, const value_type& x);
  //    template <class InputIterator>
  //        iterator insert(const_iterator position, InputIterator first, InputIterator
  //        last);
  //    template<container-compatible-range<T> R>
  //      constexpr iterator insert_range(const_iterator position, R&& rg); // C++23
  //    iterator insert(const_iterator position, initializer_list<value_type> il);
  //    iterator erase(const_iterator position);
  //    iterator erase(const_iterator first, const_iterator last);

  // HOSTDEV constexpr void
  // reserve(Size n);

  HOSTDEV constexpr void
  clear() noexcept;

  HOSTDEV constexpr void
  resize(Size n) noexcept;
  ////
  ////  HOSTDEV inline void push_back(T const & value);
  ////
  ////  HOSTDEV void insert(T const * pos, Size n, T const & value);
  ////
  ////  HOSTDEV void insert(T const * pos, T const & value);
  ////
  ////  PURE HOSTDEV [[nodiscard]] constexpr auto
  ////  contains(T const & value) const noexcept -> bool
  /// requires(!std::floating_point<T>);
  ////
  // -----------------------------------------------------------------------------
  // Operators
  // -----------------------------------------------------------------------------

  PURE HOSTDEV constexpr auto
  // cppcheck-suppress functionConst
  operator[](Size i) noexcept -> T &;

  PURE HOSTDEV constexpr auto
  operator[](Size i) const noexcept -> T const &;

  HOSTDEV constexpr auto
  operator=(Vector const & v) noexcept -> Vector &;

  HOSTDEV constexpr auto
  operator=(Vector && v) noexcept -> Vector &;
  ////
  ////  PURE HOSTDEV constexpr auto operator==(Vector const & v) const noexcept -> bool;
  //

  // -----------------------------------------------------------------------------
  // Hidden
  // -----------------------------------------------------------------------------

  HOSTDEV HIDDEN constexpr void
  allocate(Size n) noexcept;

  HOSTDEV HIDDEN constexpr void
  // NOLINTNEXTLINE(readability-identifier-naming)
  construct_at_end(Size n) noexcept;

  HOSTDEV HIDDEN constexpr void
  // NOLINTNEXTLINE(readability-identifier-naming)
  construct_at_end(Size n, T const & value) noexcept;

  HOSTDEV HIDDEN constexpr void
  // NOLINTNEXTLINE(readability-identifier-naming)
  destruct_at_end(Ptr new_last) noexcept;

  HOSTDEV HIDDEN constexpr void
  // NOLINTNEXTLINE(readability-identifier-naming)
  append_default(Size n) noexcept;

  HOSTDEV [[nodiscard]] HIDDEN constexpr auto
  recommend(Size new_size) const noexcept -> Size;

}; // struct Vector

// -----------------------------------------------------------------------------
// Methods
// -----------------------------------------------------------------------------

// template <typename T>
// requires(std::is_arithmetic_v<T> && !std::unsigned_integral<T>) PURE HOSTDEV
//     constexpr auto isApprox(Vector<T> const & a, Vector<T> const & b,
//                             T epsilon = T{}) noexcept -> bool;
//
// template <typename T>
// requires(std::unsigned_integral<T>) PURE HOSTDEV
//     constexpr auto isApprox(Vector<T> const & a, Vector<T> const & b,
//                             T epsilon = T{}) noexcept -> bool;

// Vector<bool> is a specialization that is not supported
template <>
struct Vector<bool> {
};

} // namespace um2

#include "Vector.inl"
