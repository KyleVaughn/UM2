#pragma once

#include <um2/stdlib/algorithm/copy.hpp>
#include <um2/stdlib/algorithm/equal.hpp>
#include <um2/stdlib/algorithm/lexicographical_compare.hpp>
#include <um2/stdlib/algorithm/max.hpp>
#include <um2/stdlib/assert.hpp>
#include <um2/stdlib/memory/addressof.hpp>
#include <um2/stdlib/memory/construct_at.hpp>
#include <um2/stdlib/utility/is_pointer_in_range.hpp>
#include <um2/stdlib/utility/move.hpp>
#include <um2/stdlib/utility/swap.hpp>

#include <initializer_list> // std::initializer_list

//==============================================================================
// VECTOR
//==============================================================================

namespace um2
{

template <class T>
class Vector
{
public:
  using Ptr = T *;
  using ConstPtr = T const *;

private:
  Ptr _begin = nullptr;
  Ptr _end = nullptr;
  Ptr _end_cap = nullptr;

  //==============================================================================
  // Private member functions
  //==============================================================================

  // Allocate memory for n elements
  HOSTDEV inline void
  allocate(Int n) noexcept;

  // Append n default-initialized elements to the end of the vector
  // Grows the capacity of the vector if necessary
  // Retains the values of the elements already in the vector
  HOSTDEV inline void
  append(Int n) noexcept;

  // Construct n default-initialized elements at the end of the vector
  // Assumes that there is enough capacity to hold the new elements
  HOSTDEV inline void
  constructAtEnd(Int n) noexcept;

  // Construct n elements at the end of the vector, each with value
  // Assumes that there is enough capacity to hold the new elements
  HOSTDEV inline void
  constructAtEnd(Int n, T const & value) noexcept;

  // Construct n elements at the end of the vector, copying from [first, last)
  // n disambiguates the function from the previous one
  // Assumes that there is enough capacity to hold the new elements
  template <class InputIt>
  HOSTDEV void
  constructAtEnd(InputIt first, InputIt last, Int n) noexcept;

  // Destroy the elements and deallocate the buffer
  HOSTDEV void
  deallocate() noexcept;

  // Destroy elements at the end of the vector until new_last.
  // Does not change capacity.
  // _begin <= new_last <= _end
  HOSTDEV inline void
  destructAtEnd(Ptr new_last) noexcept;

  template <class... Args>
  HOSTDEV inline auto
  emplaceBackSlowPath(Args &&... args) noexcept -> Ptr;

  template <class U>
  HOSTDEV inline auto
  pushBackSlowPath(U && value) noexcept -> Ptr;

  // Return the recommended capacity for a vector of size new_size.
  // Either double the current capacity or use the new_size if it is larger.
  PURE HOSTDEV [[nodiscard]] inline constexpr auto
  recommend(Int new_size) const noexcept -> Int;

  // Relocate the objects in the range [begin, end) into the front of v and
  // swap *this with v. It is assumed that v provides enough capacity to hold
  // the elements in the range [begin, end).
  HOSTDEV inline void
  swapBuffers(Vector & v) noexcept;

public:
  //==============================================================================
  // Constructors and assignment
  //==============================================================================

  constexpr Vector() noexcept = default;

  HOSTDEV explicit Vector(Int n) noexcept;

  HOSTDEV
  Vector(Int n, T const & value) noexcept;

  HOSTDEV
  Vector(Vector const & v) noexcept;

  HOSTDEV inline Vector(Vector && v) noexcept;

  HOSTDEV
  Vector(T const * first, T const * last) noexcept;

  inline Vector(std::initializer_list<T> const & list) noexcept;

  HOSTDEV inline auto
  operator=(Vector const & v) noexcept -> Vector &;

  HOSTDEV inline auto
  operator=(Vector && v) noexcept -> Vector &;

  auto
  operator=(std::initializer_list<T> const & list) noexcept -> Vector &;

  template <class InputIt>
  HOSTDEV void
  assign(InputIt first, InputIt last) noexcept;

  //==============================================================================
  // Destructor
  //==============================================================================

  HOSTDEV ~Vector() noexcept;

  //==============================================================================
  // Element access
  //==============================================================================

  PURE HOSTDEV inline constexpr auto
  operator[](Int i) noexcept -> T &;

  PURE HOSTDEV inline constexpr auto
  operator[](Int i) const noexcept -> T const &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  front() noexcept -> T &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  front() const noexcept -> T const &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  back() noexcept -> T &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  back() const noexcept -> T const &;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  data() noexcept -> T *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  data() const noexcept -> T const *;

  //==============================================================================
  // Iterators
  //==============================================================================

  PURE HOSTDEV [[nodiscard]] inline constexpr auto
  begin() noexcept -> T *;

  PURE HOSTDEV [[nodiscard]] inline constexpr auto
  begin() const noexcept -> T const *;

  PURE HOSTDEV [[nodiscard]] inline constexpr auto
  end() noexcept -> T *;

  PURE HOSTDEV [[nodiscard]] inline constexpr auto
  end() const noexcept -> T const *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  cbegin() const noexcept -> T const *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  cend() const noexcept -> T const *;

  //==============================================================================
  // Capacity
  //==============================================================================

  PURE HOSTDEV [[nodiscard]] constexpr auto
  empty() const noexcept -> bool;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  size() const noexcept -> Int;

  PURE HOSTDEV [[nodiscard]] static constexpr auto
  // NOLINTNEXTLINE(readability-identifier-naming) match std::vector
  max_size() noexcept -> Int;

  HOSTDEV void
  reserve(Int n) noexcept;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  capacity() const noexcept -> Int;

  //==============================================================================
  // Modifiers
  //==============================================================================

  // Doesn't change capacity
  HOSTDEV void
  clear() noexcept;

  HOSTDEV inline void
  // NOLINTNEXTLINE(readability-identifier-naming) match std::vector
  push_back(T const & value) noexcept;

  HOSTDEV inline void
  // NOLINTNEXTLINE(readability-identifier-naming) match std::vector
  push_back(T && value) noexcept;

  template <typename... Args>
  HOSTDEV inline void
  // NOLINTNEXTLINE(readability-identifier-naming) match std::vector
  emplace_back(Args &&... args) noexcept;

  HOSTDEV void
  resize(Int n) noexcept;

}; // class Vector

// Vector<bool> is a specialization that is not supported
template <>
class Vector<bool>
{
};

//==============================================================================
// Relational operators
//==============================================================================

template <class T>
HOSTDEV constexpr auto
operator==(Vector<T> const & l, Vector<T> const & r) noexcept -> bool;

template <class T>
HOSTDEV constexpr auto
operator<(Vector<T> const & l, Vector<T> const & r) noexcept -> bool;

template <class T>
HOSTDEV constexpr auto
operator!=(Vector<T> const & l, Vector<T> const & r) noexcept -> bool;

template <class T>
HOSTDEV constexpr auto
operator>(Vector<T> const & l, Vector<T> const & r) noexcept -> bool;

template <class T>
HOSTDEV constexpr auto
operator<=(Vector<T> const & l, Vector<T> const & r) noexcept -> bool;

template <class T>
HOSTDEV constexpr auto
operator>=(Vector<T> const & l, Vector<T> const & r) noexcept -> bool;

//==============================================================================
// Private member functions
//==============================================================================

// Allocate memory for n elements
template <class T>
HOSTDEV inline void
Vector<T>::allocate(Int n) noexcept
{
  ASSERT(0 < n);
  ASSERT(n < max_size());
  ASSERT(_begin == nullptr);
  _begin = static_cast<T *>(::operator new(static_cast<size_t>(n) * sizeof(T)));
  _end = _begin;
  _end_cap = _begin + n;
}

// Construct n default-initialized elements at the end of the vector
template <class T>
HOSTDEV inline void
Vector<T>::constructAtEnd(Int n) noexcept
{
  Ptr new_end = _end + n;
  for (; _end != new_end; ++_end) {
    um2::construct_at(_end);
  }
}

// Construct n elements with value at the end of the vector
template <class T>
HOSTDEV inline void
Vector<T>::constructAtEnd(Int n, T const & value) noexcept
{
  Ptr new_end = _end + n;
  for (; _end != new_end; ++_end) {
    um2::construct_at(_end, value);
  }
}

// Construct n elements at the end of the vector, copying from [first, last)
// n disambiguates the function from the previous one, but is not used when
// asserts are disabled
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
template <class T>
template <class InputIt>
HOSTDEV inline void
// NOLINTNEXTLINE(clang-diagnostic-unused-parameter,misc-unused-parameters)
Vector<T>::constructAtEnd(InputIt first, InputIt last, Int n) noexcept
{
  ASSERT(n >= 0);
  ASSERT(n == static_cast<Int>(last - first));
  for (; first != last; ++_end, ++first) {
    um2::construct_at(_end, *first);
  }
}
#pragma GCC diagnostic pop

// Destroy the elements and deallocate the buffer
template <class T>
HOSTDEV inline void
Vector<T>::deallocate() noexcept
{
  if (_begin != nullptr) {
    clear();
    ::operator delete(_begin);
    _begin = nullptr;
    _end = nullptr;
    _end_cap = nullptr;
  }
}

// Destroy elements at the end of the vector until new_last
template <class T>
HOSTDEV inline void
Vector<T>::destructAtEnd(Ptr new_last) noexcept
{
  Ptr soon_to_be_end = _end;
  while (new_last != soon_to_be_end) {
    um2::destroy_at(--soon_to_be_end);
  }
  _end = new_last;
}

template <class T>
template <class... Args>
HOSTDEV inline auto
Vector<T>::emplaceBackSlowPath(Args &&... args) noexcept -> Ptr
{
  Vector<T> temp;
  temp.allocate(recommend(size() + 1));
  swapBuffers(temp);
  um2::construct_at(_end, um2::forward<Args>(args)...);
  ++_end;
  return _end;
}

template <class T>
template <class U>
HOSTDEV inline auto
Vector<T>::pushBackSlowPath(U && value) noexcept -> Ptr
{
  Vector<T> temp;
  temp.allocate(recommend(size() + 1));
  swapBuffers(temp);
  um2::construct_at(_end, um2::forward<U>(value));
  ++_end;
  return _end;
}

// Return the recommended capacity for a new size. Either double the current
// capacity or use the new size if it is larger.
template <class T>
PURE HOSTDEV constexpr auto
Vector<T>::recommend(Int new_size) const noexcept -> Int
{
  return um2::max(2 * capacity(), new_size);
}

template <class T>
HOSTDEV inline void
Vector<T>::swapBuffers(Vector & v) noexcept
{
  // ASSUMES v IS UNINITIALIZED IN [begin, end)
  // v may have initialized objects in [end, end_cap), but _end will only be
  // updated to reflect having size() objects.

  ASSERT(v._begin != nullptr);
  ASSERT(v.empty());

  // Move the objects in the range [first, last) into the front of v
  Ptr pold = _begin;
  Ptr pnew = v._begin;
  // if T is trivially move-constructible and trivially destructible, then we
  // can use memcpy to move the objects
  if constexpr (std::is_trivially_move_constructible_v<T> &&
                std::is_trivially_destructible_v<T>) {
    memcpy(pnew, pold, static_cast<size_t>(size()) * sizeof(T));
  } else {
    for (; pold != _end; ++pold, ++pnew) {
      um2::construct_at(pnew, um2::move(*pold));
    }
  }
  // Swap the pointers
  um2::swap(_begin, v._begin);
  um2::swap(_end, v._end);
  um2::swap(_end_cap, v._end_cap);
  // Make sure _end reflects the new end of the vector
  _end = _begin + (v._end - v._begin);
  // v now contains the old buffer, which has invalidated objects due
  // to the move operations. The destructor of v will take care of destroying
  // the objects and deallocating the buffer.
  ASSERT(size() == v.size());
}

// Append n default-initialized elements to the end of the vector
template <class T>
HOSTDEV inline void
Vector<T>::append(Int n) noexcept
{
  if (static_cast<Int>(_end_cap - _end) < n) {
    Vector<T> temp;
    temp.allocate(recommend(size() + n));
    swapBuffers(temp);
  }
  constructAtEnd(n);
}

//==============================================================================
// Constructors and assignment
//==============================================================================

// Default construct n elements
template <class T>
HOSTDEV
Vector<T>::Vector(Int const n) noexcept
{
  ASSERT(n > 0);
  this->allocate(n);
  constructAtEnd(n);
}

// Construct n elements with value
template <class T>
HOSTDEV
Vector<T>::Vector(Int const n, T const & value) noexcept
{
  ASSERT(n > 0);
  this->allocate(n);
  constructAtEnd(n, value);
}

// Copy construct from a vector
template <class T>
HOSTDEV
Vector<T>::Vector(Vector<T> const & v) noexcept
{
  Int const n = v.size();
  if (n == 0) {
    return;
  }
  this->allocate(n);
  constructAtEnd(n);
  um2::copy(v._begin, v._end, _begin);
}

// Move construct from a vector
template <class T>
HOSTDEV
Vector<T>::Vector(Vector<T> && v) noexcept
    : _begin{v._begin},
      _end{v._end},
      _end_cap{v._end_cap}
{
  v._begin = nullptr;
  v._end = nullptr;
  v._end_cap = nullptr;
}

template <class T>
HOSTDEV
Vector<T>::Vector(T const * first, T const * last) noexcept
{
  Int const n = static_cast<Int>(last - first);
  if (n == 0) {
    return;
  }
  this->allocate(n);
  constructAtEnd(n);
  // Check for aliasing
  ASSERT(!um2::is_pointer_in_range(first, last, _begin));
  um2::copy(first, last, _begin);
}

// Construct from an initializer list
template <class T>
Vector<T>::Vector(std::initializer_list<T> const & list) noexcept
{
  Int const n = static_cast<Int>(list.size());
  ASSERT(n > 0);
  this->allocate(n);
  constructAtEnd(n);
  um2::copy(list.begin(), list.end(), _begin);
}

template <class T>
HOSTDEV inline auto
Vector<T>::operator=(Vector<T> const & v) noexcept -> Vector<T> &
{
  if (this != addressof(v)) {
    assign(v._begin, v._end);
  }
  return *this;
}

template <class T>
HOSTDEV inline auto
Vector<T>::operator=(Vector<T> && v) noexcept -> Vector<T> &
{
  if (this != addressof(v)) {
    deallocate();
    // Move the buffer from v
    _begin = v._begin;
    _end = v._end;
    _end_cap = v._end_cap;
    v._begin = nullptr;
    v._end = nullptr;
    v._end_cap = nullptr;
  }
  return *this;
}

template <class T>
auto
Vector<T>::operator=(std::initializer_list<T> const & list) noexcept -> Vector &
{
  assign(list.begin(), list.end());
  return *this;
}

// Assign from a range [first, last), adjusting the size and capacity as needed
template <class T>
template <class InputIt>
HOSTDEV void
Vector<T>::assign(InputIt first, InputIt last) noexcept
{
  // If [first, last) overlaps with the vector, we may have incorrect behavior.
  // Since first and last are InputIterators and not pointers, we cannot check
  // for overlap. :(
  Int const new_size = static_cast<Int>(last - first);
  if (new_size <= capacity()) {
    if (new_size > size()) {
      // Overwrite the existing elements
      InputIt mid = first + size();
      um2::copy(first, mid, _begin);
      // Construct the new elements
      constructAtEnd(mid, last, new_size - size());
    } else {
      T * m = um2::copy(first, last, _begin);
      destructAtEnd(m);
    }
  } else {
    this->deallocate();
    allocate(recommend(new_size));
    constructAtEnd(first, last, new_size);
  }
}

//===============================================================================
// Destructor
//===============================================================================

template <class T>
HOSTDEV Vector<T>::~Vector() noexcept
{
  // If the vector is not empty, destroy the elements and deallocate the buffer
  if (_begin != nullptr) {
    clear();
    ::operator delete(_begin);
  }
}

//===============================================================================
// Element access
//===============================================================================

template <class T>
PURE HOSTDEV inline constexpr auto
Vector<T>::operator[](Int const i) noexcept -> T &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT(i < size());
  return _begin[i];
}

template <class T>
PURE HOSTDEV inline constexpr auto
Vector<T>::operator[](Int const i) const noexcept -> T const &
{
  ASSERT_ASSUME(0 <= i);
  ASSERT(i < size());
  return _begin[i];
}

template <class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T>::front() noexcept -> T &
{
  return *_begin;
}

template <class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T>::front() const noexcept -> T const &
{
  return *_begin;
}

template <class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T>::back() noexcept -> T &
{
  ASSERT(size() > 0);
  return *(_end - 1);
}

template <class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T>::back() const noexcept -> T const &
{
  ASSERT(size() > 0);
  return *(_end - 1);
}

template <class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T>::data() noexcept -> T *
{
  return _begin;
}

template <class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T>::data() const noexcept -> T const *
{
  return _begin;
}

//==============================================================================
// Iterators
//==============================================================================

template <class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T>::begin() noexcept -> T *
{
  return _begin;
}

template <class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T>::begin() const noexcept -> T const *
{
  return _begin;
}

template <class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T>::end() noexcept -> T *
{
  return _end;
}

template <class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T>::end() const noexcept -> T const *
{
  return _end;
}

template <class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T>::cbegin() const noexcept -> T const *
{
  return _begin;
}

template <class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T>::cend() const noexcept -> T const *
{
  return _end;
}

//==============================================================================
// Capacity
//==============================================================================

template <class T>
PURE HOSTDEV constexpr auto
Vector<T>::empty() const noexcept -> bool
{
  return _begin == _end;
}

template <class T>
PURE HOSTDEV constexpr auto
Vector<T>::size() const noexcept -> Int
{
  return static_cast<Int>(_end - _begin);
}

// Return the maximum number of elements the vector can hold
template <class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T>::max_size() noexcept -> Int
{
  return intMax() / sizeof(T);
}

template <class T>
HOSTDEV void
Vector<T>::reserve(Int const n) noexcept
{
  if (n > capacity()) {
    Vector<T> temp;
    temp.allocate(n);
    swapBuffers(temp);
  }
}

template <class T>
PURE HOSTDEV constexpr auto
Vector<T>::capacity() const noexcept -> Int
{
  return static_cast<Int>(_end_cap - _begin);
}

//===============================================================================
// Modifiers
//===============================================================================

// Does not change capacity
template <class T>
HOSTDEV void
Vector<T>::clear() noexcept
{
  destructAtEnd(_begin);
}

template <class T>
HOSTDEV inline void
Vector<T>::push_back(T const & value) noexcept
{
  if (_end < _end_cap) {
    um2::construct_at(_end, value);
    ++_end;
  } else [[unlikely]] {
    _end = pushBackSlowPath(value);
  }
}

template <class T>
HOSTDEV inline void
Vector<T>::push_back(T && value) noexcept
{
  if (_end < _end_cap) {
    um2::construct_at(_end, um2::move(value));
    ++_end;
  } else [[unlikely]] {
    _end = pushBackSlowPath(um2::move(value));
  }
}

template <class T>
template <class... Args>
HOSTDEV inline void
Vector<T>::emplace_back(Args &&... args) noexcept
{
  if (_end < _end_cap) {
    um2::construct_at(_end, um2::forward<Args>(args)...);
    ++_end;
  } else [[unlikely]] {
    _end = emplaceBackSlowPath(um2::forward<Args>(args)...);
  }
}

template <class T>
HOSTDEV void
Vector<T>::resize(Int const n) noexcept
{
  Int const cs = size();
  // If we are shrinking, destroy the elements that are no longer needed
  // If we are growing, default construct the new elements
  if (cs < n) {
    append(n - cs);
  } else if (cs > n) {
    destructAtEnd(_begin + n);
  }
}

//==============================================================================
// Relational operators
//==============================================================================

template <class T>
HOSTDEV constexpr auto
operator==(Vector<T> const & l, Vector<T> const & r) noexcept -> bool
{
  return l.size() == r.size() && um2::equal(l.cbegin(), l.cend(), r.cbegin());
}

template <class T>
HOSTDEV constexpr auto
operator<(Vector<T> const & l, Vector<T> const & r) noexcept -> bool
{
  return um2::lexicographical_compare(l.cbegin(), l.cend(), r.cbegin(), r.cend());
}

template <class T>
HOSTDEV constexpr auto
operator!=(Vector<T> const & l, Vector<T> const & r) noexcept -> bool
{
  return !(l == r);
}

template <class T>
HOSTDEV constexpr auto
operator>(Vector<T> const & l, Vector<T> const & r) noexcept -> bool
{
  return r < l;
}

template <class T>
HOSTDEV constexpr auto
operator<=(Vector<T> const & l, Vector<T> const & r) noexcept -> bool
{
  return !(r < l);
}

template <class T>
HOSTDEV constexpr auto
operator>=(Vector<T> const & l, Vector<T> const & r) noexcept -> bool
{
  return !(l < r);
}

} // namespace um2
