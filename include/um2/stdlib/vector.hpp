#pragma once


#include <um2/stdlib/algorithm/copy.hpp>
#include <um2/stdlib/algorithm/max.hpp>
#include <um2/stdlib/assert.hpp>
#include <um2/stdlib/memory/addressof.hpp>
#include <um2/stdlib/memory/construct_at.hpp>
#include <um2/stdlib/utility/move.hpp>
#include <um2/stdlib/utility/swap.hpp>

#include <initializer_list> // std::initializer_list

//==============================================================================
// VECTOR
//==============================================================================
// An std::vector-like class without and Allocator template parameter.

namespace um2
{

template <class T>
class Vector
{

  using Ptr = T *;
  using ConstPtr = T const *;

  Ptr _begin = nullptr;
  Ptr _end = nullptr;
  Ptr _end_cap = nullptr;

  //==============================================================================
  // Private member functions 
  //==============================================================================
  // NOLINTBEGIN(readability-identifier-naming) match std::vector

  // Allocate memory for n elements
  HOSTDEV constexpr void
  allocate(Int n) noexcept;

  // Append n default-initialized elements to the end of the vector
  // Grows the capacity of the vector if necessary
  // Retains the values of the elements already in the vector
  HOSTDEV constexpr void
  append(Int n) noexcept;

  // Assign from a range [first, last), adjusting the size and capacity as needed
  template <class InputIt>
  HOSTDEV constexpr void
  assign(InputIt first, InputIt last) noexcept;

  // Construct n default-initialized elements at the end of the vector
  HOSTDEV constexpr void
  construct_at_end(Int n) noexcept;

  // Construct n elements at the end of the vector, each with value
  HOSTDEV constexpr void
  construct_at_end(Int n, T const & value) noexcept;

  // Construct n elements at the end of the vector, copying from [first, last)
  // Yes, n is redundant.
  template <class InputIt>
  HOSTDEV constexpr void
  construct_at_end(InputIt first, InputIt last, Int n) noexcept;

  // Destroy the elements and deallocate the buffer
  HOSTDEV constexpr void
  deallocate() noexcept;

  // Destroy elements at the end of the vector until new_last.
  // Does not change capacity.
  // _begin <= new_last <= _end
  HOSTDEV constexpr void
  destruct_at_end(Ptr new_last) noexcept;

  template <class... Args>
  HOSTDEV constexpr auto
  emplace_back_slow_path(Args &&... args) noexcept -> Ptr;

  template <class U>
  HOSTDEV constexpr auto
  push_back_slow_path(U && value) noexcept -> Ptr;

  // Return the recommended capacity for a vector of size new_size.
  // Either double the current capacity or use the new_size if it is larger.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  recommend(Int new_size) const noexcept -> Int;

  // Relocate the objects in the range [begin, end) into the front of v and
  // swap *this with v. It is assumed that v provides enough capacity to hold
  // the elements in the range [begin, end).
  HOSTDEV constexpr void
  swap_buffers(Vector & v) noexcept;

public:
  //==============================================================================
  // Constructors
  //==============================================================================

  constexpr Vector() noexcept = default;

  HOSTDEV explicit constexpr Vector(Int n) noexcept;

  HOSTDEV constexpr Vector(Int n, T const & value) noexcept;

  HOSTDEV constexpr Vector(Vector const & v) noexcept;

  HOSTDEV constexpr Vector(Vector && v) noexcept;

  HOSTDEV constexpr Vector(T const * first, T const * last) noexcept;

  constexpr Vector(std::initializer_list<T> const & list) noexcept;

  //==============================================================================
  // Destructor
  //==============================================================================

  HOSTDEV constexpr ~Vector() noexcept;

  //==============================================================================
  // Accessors
  //==============================================================================

  PURE HOSTDEV [[nodiscard]] static constexpr auto
  max_size() noexcept -> Int;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  begin() noexcept -> T *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  begin() const noexcept -> T const *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  end() noexcept -> T *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  end() const noexcept -> T const *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  size() const noexcept -> Int;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  capacity() const noexcept -> Int;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  empty() const noexcept -> bool;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  cbegin() const noexcept -> T const *;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  cend() const noexcept -> T const *;

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
  // Member functions
  //==============================================================================

  HOSTDEV constexpr void
  clear() noexcept;

  HOSTDEV constexpr void
  resize(Int n) noexcept;

  HOSTDEV constexpr void
  reserve(Int n) noexcept;

  HOSTDEV inline constexpr void
  push_back(T const & value) noexcept;

  HOSTDEV inline constexpr void
  push_back(T && value) noexcept;

//  HOSTDEV constexpr void
//  push_back(Int n, T const & value) noexcept;

  template <typename... Args>
  HOSTDEV constexpr void
  emplace_back(Args &&... args) noexcept;

//  HOSTDEV constexpr void
//  pop_back() noexcept;
//
  //==============================================================================
  // Operators
  //==============================================================================

  PURE HOSTDEV inline constexpr auto
  operator[](Int i) noexcept -> T &;

  PURE HOSTDEV inline constexpr auto
  operator[](Int i) const noexcept -> T const &;

  HOSTDEV inline constexpr auto
  operator=(Vector const & v) noexcept -> Vector &;

  HOSTDEV inline constexpr auto
  operator=(Vector && v) noexcept -> Vector &;

  constexpr auto
  operator=(std::initializer_list<T> const & list) noexcept -> Vector &;

//  PURE constexpr auto
//  operator==(Vector const & v) const noexcept -> bool;
//
  // NOLINTEND(readability-identifier-naming)
}; // class Vector

// Vector<bool> is a specialization that is not supported
template <>
class Vector<bool>
{
};

//==============================================================================
// Private methods
//==============================================================================

// Allocate memory for n elements
template <class T>
HOSTDEV constexpr void
Vector<T>::allocate(Int n) noexcept
{
  ASSERT(0 < n);
  ASSERT(n < max_size());
  ASSERT(_begin == nullptr);
  _begin = static_cast<T *>(::operator new(static_cast<size_t>(n) * sizeof(T)));
  _end = _begin;
  _end_cap = _begin + n;
}

// Assign from a range [first, last), adjusting the size and capacity as needed
template <class T>
template <class InputIt>
HOSTDEV constexpr void
Vector<T>::assign(InputIt first, InputIt last) noexcept
{
  Int const new_size = static_cast<Int>(last - first);
  if (new_size <= capacity()) {
    if (new_size > size()) {
      // Overwrite the existing elements
      InputIt mid = first + size();
      um2::copy(first, mid, _begin);
      // Construct the new elements
      construct_at_end(mid, last, new_size - size());
    } else {
      InputIt m = um2::copy(first, last, _begin); 
      destruct_at_end(m);
    }
  } else {
    this->deallocate();
    allocate(recommend(new_size));
    construct_at_end(first, last, new_size);
  }
}

// Construct n default-initialized elements at the end of the vector
template <class T>
HOSTDEV constexpr void
Vector<T>::construct_at_end(Int n) noexcept
{
  Ptr new_end = _end + n;
  for (; _end != new_end; ++_end) {
    um2::construct_at(_end);
  }
}

// Construct n elements with value at the end of the vector
template <class T>
HOSTDEV constexpr void
Vector<T>::construct_at_end(Int n, T const & value) noexcept
{
  Ptr new_end = _end + n;
  for (; _end != new_end; ++_end) {
    um2::construct_at(_end, value);
  }
}

// Construct n elements at the end of the vector, copying from [first, last)
template <class T>
template <class InputIt>
HOSTDEV constexpr void
Vector<T>::construct_at_end(InputIt first, InputIt last, Int n) noexcept
{
  ASSERT(n >= 0);
  ASSERT(n == static_cast<Int>(last - first));
  for (; first != last; ++_end, ++first) {
    um2::construct_at(_end, *first);
  }
}

// Destroy the elements and deallocate the buffer
template <class T>
HOSTDEV constexpr void
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
HOSTDEV constexpr void
Vector<T>::destruct_at_end(Ptr new_last) noexcept
{
  Ptr soon_to_be_end = _end;
  while (new_last != soon_to_be_end) {
    um2::destroy_at(--soon_to_be_end);
  }
  _end = new_last;
}

template <class T>
template <class... Args>
HOSTDEV constexpr auto
Vector<T>::emplace_back_slow_path(Args &&... args) noexcept -> Ptr
{
  Vector<T> temp;
  temp.allocate(recommend(size() + 1));
  swap_buffers(temp);
  um2::construct_at(_end, um2::forward<Args>(args)...);
  ++_end;
  return _end;
}

template <class T>
template <class U>
HOSTDEV constexpr auto
Vector<T>::push_back_slow_path(U && value) noexcept -> Ptr
{
  Vector<T> temp;
  temp.allocate(recommend(size() + 1));
  swap_buffers(temp);
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
HOSTDEV constexpr void
Vector<T>::swap_buffers(Vector & v) noexcept
{
  // ASSUMES v IS UNINITIALIZED IN [begin, end)
  // v may have initialized objects in [end, end_cap), but _end will only be
  // updated to reflect having size() objects.


  // This means we can use the move constructor to move the objects instead
  // of the move assignment operator.
  //

  ASSERT(v._begin != nullptr);
  ASSERT(v.empty());

  // Move the objects in the range [first, last) into the front of v
  Ptr pold = _begin;
  Ptr pnew = v._begin;
  for (; pold != _end; ++pold, ++pnew) {
    um2::construct_at(pnew, um2::move(*pold));
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
HOSTDEV constexpr void
Vector<T>::append(Int n) noexcept
{
  if (static_cast<Int>(_end_cap - _end) < n) {
    Vector<T> temp;
    temp.allocate(recommend(size() + n));
    swap_buffers(temp);
  }
  construct_at_end(n);
}

//==============================================================================
// Constructors
//==============================================================================

// Default construct n elements
template <class T>
HOSTDEV constexpr Vector<T>::Vector(Int const n) noexcept
{
  this->allocate(n);
  construct_at_end(n);
}

// Construct n elements with value
template <class T>
HOSTDEV constexpr Vector<T>::Vector(Int const n, T const & value) noexcept
{
  this->allocate(n);
  construct_at_end(n, value);
}

// Copy construct from a vector
template <class T>
HOSTDEV constexpr Vector<T>::Vector(Vector<T> const & v) noexcept
{
  Int const n = v.size();
  this->allocate(n);
  construct_at_end(n);
  um2::copy(v._begin, v._end, _begin);
}

// Move construct from a vector
template <class T>
HOSTDEV constexpr Vector<T>::Vector(Vector<T> && v) noexcept
    : _begin{v._begin},
      _end{v._end},
      _end_cap{v._end_cap}
{
  v._begin = nullptr;
  v._end = nullptr;
  v._end_cap = nullptr;
}

template <class T>
HOSTDEV constexpr Vector<T>::Vector(T const * first, T const * last) noexcept
{
  Int const n = static_cast<Int>(last - first);
  this->allocate(n);
  construct_at_end(n);
  um2::copy(first, last, _begin);
}

// Construct from an initializer list
template <class T>
constexpr Vector<T>::Vector(std::initializer_list<T> const & list) noexcept
{
  // Initializer lists can't be moved from, so we have to copy.
  Int const n = static_cast<Int>(list.size());
  this->allocate(n);
  construct_at_end(n);
  um2::copy(list.begin(), list.end(), _begin);
}

//==============================================================================-
// Destructor
//==============================================================================-

template <class T>
HOSTDEV constexpr Vector<T>::~Vector() noexcept
{
  // If the vector is not empty, destroy the elements and deallocate the buffer
  if (_begin != nullptr) {
    clear();
    ::operator delete(_begin);
  }
}

//==============================================================================-
// Accessors
//==============================================================================-

// Return the maximum number of elements the vector can hold
template <class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T>::max_size() noexcept -> Int
{
  return intMax() / sizeof(T);
}

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
PURE HOSTDEV constexpr auto
Vector<T>::size() const noexcept -> Int
{
  return static_cast<Int>(_end - _begin);
}

template <class T>
PURE HOSTDEV constexpr auto
Vector<T>::capacity() const noexcept -> Int
{
  return static_cast<Int>(_end_cap - _begin);
}

template <class T>
PURE HOSTDEV constexpr auto
Vector<T>::empty() const noexcept -> bool
{
  return _begin == _end;
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

//==============================================================================-
// Operators
//==============================================================================-

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
HOSTDEV inline constexpr auto
Vector<T>::operator=(Vector<T> const & v) noexcept -> Vector<T> &
{
  if (this != addressof(v)) {
    assign(v._begin, v._end);
  }
  return *this;
}

template <class T>
HOSTDEV inline constexpr auto
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
constexpr auto
Vector<T>::operator=(std::initializer_list<T> const & list) noexcept -> Vector &
{
  assign(list.cbegin(), list.cend());
  return *this;
}

//template <class T>
//PURE constexpr auto
//Vector<T>::operator==(Vector<T> const & v) const noexcept -> bool
//{
//  return size() == v.size() && std::equal(begin(), end(), v.begin());
//}
//
//==============================================================================
// Member functions 
//==============================================================================

// Does not change capacity
template <class T>
HOSTDEV constexpr void
Vector<T>::clear() noexcept
{
  destruct_at_end(_begin);
}

template <class T>
HOSTDEV constexpr void
Vector<T>::resize(Int const n) noexcept
{
  Int const cs = size();
  // If we are shrinking, destroy the elements that are no longer needed
  // If we are growing, default construct the new elements
  if (cs < n) {
    append(n - cs);
  } else if (cs > n) {
    destruct_at_end(_begin + n);
  }
}

template <class T>
HOSTDEV constexpr void
Vector<T>::reserve(Int const n) noexcept
{
  if (n > capacity()) {
    Vector<T> temp;
    temp.allocate(n);
    swap_buffers(temp);
  }
}

template <class T>
HOSTDEV inline constexpr void
Vector<T>::push_back(T const & value) noexcept
{
  if (_end < _end_cap) {
    um2::construct_at(_end, value);
    ++_end;
  } else {
    _end = push_back_slow_path(value);
  }
}

template <class T>
HOSTDEV inline constexpr void
Vector<T>::push_back(T && value) noexcept
{
  if (_end < _end_cap) {
    um2::construct_at(_end, um2::move(value));
    ++_end;
  } else {
    _end = push_back_slow_path(um2::move(value));
  }
}

//template <class T>
//HOSTDEV constexpr void
//Vector<T>::push_back(Int const n, T const & value) noexcept
//{
//  // If we have enough capacity, just construct the new elements
//  // Otherwise, this->allocate a new buffer and move the elements over
//  if (static_cast<Int>(_end_cap - _end) < n) {
//    grow(n);
//  }
//  // Construct the new elements
//  construct_at_end(n, value);
//}

template <class T>
template <class... Args>
HOSTDEV inline constexpr void
Vector<T>::emplace_back(Args &&... args) noexcept
{
  if (_end < _end_cap) {
    um2::construct_at(_end, um2::forward<Args>(args)...);
    ++_end;
  } else {
    _end = emplace_back_slow_path(um2::forward<Args>(args)...);
  }
}

//template <class T>
//HOSTDEV constexpr void
//Vector<T>::pop_back() noexcept
//{
//  ASSERT(size() > 0);
//  um2::destroy_at(--_end);
//}

} // namespace um2
