namespace um2
{
// ---------------------------------------------------------------------------
// Constructors
// ---------------------------------------------------------------------------

template <class T, class Allocator>
HOSTDEV constexpr Vector<T, Allocator>::Vector(Allocator const & a) noexcept
    : _end_cap(nullptr, a)
{
}

template <class T, class Allocator>
HOSTDEV
Vector<T, Allocator>::Vector(Size const n)
{
  assert(n > 0);
  allocate(n);
  constructAtEnd(n);
}
//
// template <class T>
// HOSTDEV Vector<T>::Vector(Size const n, T const & value)
//     : _size{n},
//       _capacity{bit_ceil(n)},
//       _data{new T[static_cast<uSize>(bit_ceil(n))]}
//{
//   assert(n > 0);
//   for (Size i = 0; i < n; ++i) {
//     _data[i] = value;
//   }
// }
//
template <class T, class Allocator>
HOSTDEV constexpr Vector<T, Allocator>::Vector(Vector<T, Allocator> const & v)
 : _end_cap(nullptr, v.getAllocator())
{
  initWithSize(v._begin, v._end, v.size());
}

// template <class T>
// HOSTDEV Vector<T>::Vector(Vector<T> && v) noexcept
//     : _size{v._size},
//       _capacity{v._capacity},
//       _data{v._data}
//{
//   v._size = 0;
//   v._capacity = 0;
//   v._data = nullptr;
// }
//
// template <class T>
// Vector<T>::Vector(std::initializer_list<T> const & list)
//     : _size{static_cast<Size>(list.size())},
//       _capacity{static_cast<Size>(bit_ceil(list.size()))},
//       _data{new T[bit_ceil(list.size())]}
//{
//   if constexpr (std::is_trivially_copyable_v<T>) {
//     memcpy(_data, list.begin(), list.size() * sizeof(T));
//   } else {
//     Size i = 0;
//     for (auto const & value : list) {
//       _data[i++] = value;
//     }
//   }
// }
//

template <class T, class Allocator>
HOSTDEV constexpr Vector<T, Allocator>::~Vector() noexcept
{
  destroy_vector (*this)();
}

// ---------------------------------------------------------------------------
// Accessors
// ---------------------------------------------------------------------------

template <class T, class Allocator>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T, Allocator>::getAllocator() const noexcept -> Allocator
{
  return _end_cap.second;
}

template <class T, class Allocator>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T, Allocator>::begin() noexcept -> T *
{
  return _begin;
}

template <class T, class Allocator>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T, Allocator>::begin() const noexcept -> T const *
{
  return _begin;
}

template <class T, class Allocator>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T, Allocator>::end() noexcept -> T *
{
  return _end;
}

template <class T, class Allocator>
PURE HOSTDEV constexpr auto
Vector<T, Allocator>::size() const noexcept -> Size
{
  return static_cast<Size>(_end - _begin);
}

template <class T, class Allocator>
PURE HOSTDEV constexpr auto
Vector<T, Allocator>::capacity() const noexcept -> Size
{
  return static_cast<Size>(_end_cap.first - _begin);
}

template <class T, class Allocator>
PURE HOSTDEV constexpr auto
Vector<T, Allocator>::empty() const noexcept -> bool
{
  return _begin == _end;
}

template <class T, class Allocator>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T, Allocator>::cbegin() const noexcept -> T const *
{
  return begin();
}

template <class T, class Allocator>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T, Allocator>::cend() const noexcept -> T const *
{
  return end();
}

template <class T, class Allocator>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T, Allocator>::front() noexcept -> T &
{
  return *_begin;
}

template <class T, class Allocator>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T, Allocator>::front() const noexcept -> T const &
{
  return *_begin;
}

template <class T, class Allocator>
NDEBUG_PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T, Allocator>::back() noexcept -> T &
{
  assert(size() > 0);
  return *(_end - 1);
}

template <class T, class Allocator>
NDEBUG_PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T, Allocator>::back() const noexcept -> T const &
{
  assert(size() > 0);
  return *(_end - 1);
}

template <class T, class Allocator>
// cppcheck-suppress functionConst
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T, Allocator>::data() noexcept -> T *
{
  return _begin;
}

template <class T, class Allocator>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T, Allocator>::data() const noexcept -> T const *
{
  return _begin;
}

// ---------------------------------------------------------------------------
// Operators
// ---------------------------------------------------------------------------

template <class T, class Allocator>
NDEBUG_PURE HOSTDEV constexpr auto
Vector<T, Allocator>::operator[](Size const i) noexcept -> T &
{
  assert(i < size());
  return _begin[i];
}

template <class T, class Allocator>
NDEBUG_PURE HOSTDEV constexpr auto
Vector<T, Allocator>::operator[](Size const i) const noexcept -> T const &
{
  assert(i < size());
  return _begin[i];
}
// template <class T>
// HOSTDEV auto Vector<T>::operator=(Vector<T> const & v) -> Vector<T> &
//{
//  if (this != &v) {
//    if (_capacity < v.size()) {
//      delete[] _data;
//      _data = new T[static_cast<uSize>(bit_ceil(v.size()))];
//      _capacity = bit_ceil(v.size());
//    }
//    _size = v.size();
//    if constexpr (std::is_trivially_copyable_v<T>) {
//      memcpy(_data, v._data, static_cast<Size>(v.size()) * sizeof(T));
//    } else {
//      for (Size i = 0; i < v.size(); ++i) {
//        _data[i] = v._data[i];
//      }
//    }
//  }
//  return *this;
//}
//
// template <class T>
// HOSTDEV auto Vector<T>::operator=(Vector<T> && v) noexcept -> Vector<T> &
//{
//  if (this != &v) {
//    delete[] _data;
//    _size = v._size;
//    _capacity = v._capacity;
//    _data = v._data;
//    v._size = 0;
//    v._capacity = 0;
//    v._data = nullptr;
//  }
//  return *this;
//}
//
// template <class T>
// PURE HOSTDEV constexpr auto
// Vector<T>::operator==(Vector<T> const & v) const noexcept -> bool
//{
//  if (_size != v._size) {
//    return false;
//  }
//  for (Size i = 0; i < _size; ++i) {
//    if (_data[i] != v._data[i]) {
//      return false;
//    }
//  }
//  return true;
//}
//
// ---------------------------------------------------------------------------
// Methods
// ---------------------------------------------------------------------------

// template <class T, class Allocator>
// constexpr void
// Vector<T, Allocator>::reserve(Size n)
//{
//     if (n > capacity()) {
//         if (n > max_size())
//             this->__throw_length_error();
//         allocator_type& __a = this->__alloc();
//         __split_buffer<value_type, allocator_type&> __v(__n, size(), __a);
//         __swap_out_circular_buffer(__v);
//     }
// }

// template <class T>
// HOSTDEV void Vector<T>::clear() noexcept
//{
//  _size = 0;
//  _capacity = 0;
//  delete[] _data;
//  _data = nullptr;
//}
//
// template <class T>
// HOSTDEV inline void Vector<T>::reserve(Size n)
//{
//  if (_capacity < n) {
//    // since n > 0, we are safe to cast to unsigned and use bit_ceil
//    // to determine the next power of 2
//    n = bit_ceil(n);
//    T * new_data = new T[static_cast<Size>(n)];
//    if constexpr (std::is_trivially_copyable_v<T>) {
//      memcpy(new_data, _data, static_cast<Size>(_size) * sizeof(T));
//    } else {
//      for (Size i = 0; i < _size; ++i) {
//        new_data[i] = _data[i];
//      }
//    }
//    delete[] _data;
//    _data = new_data;
//    _capacity = n;
//  }
//}
//
// template <class T>
// HOSTDEV void Vector<T>::resize(Size const n)
//{
//  reserve(n);
//  _size = n;
//}
//
// template <class T>
// HOSTDEV inline void Vector<T>::push_back(T const & value)
//{
//  reserve(_size + 1);
//  _data[_size++] = value;
//}
//
// template <class T>
// PURE HOSTDEV constexpr auto Vector<T>::empty() const -> bool
//{
//  return _size == 0;
//}
//
// template <class T>
// HOSTDEV void Vector<T>::insert(T const * pos, Size const n, T const & value)
//{
//  if (n == 0) {
//    return;
//  }
//  auto const offset = static_cast<Size>(pos - _data);
//  assert(0 <= offset && offset <= _size);
//  Size const new_size = _size + n;
//  reserve(new_size);
//  // Shift elements to make room for the insertion
//  for (Size i = _size - 1; i >= offset; --i) {
//    _data[i + n] = _data[i];
//  }
//  // Insert elements
//  for (Size i = offset; i < offset + n; ++i) {
//    _data[i] = value;
//  }
//  _size = new_size;
//}
//
// template <class T>
// HOSTDEV void Vector<T>::insert(T const * pos, T const & value)
//{
//  insert(pos, 1, value);
//}
//
// template <class T>
// PURE HOSTDEV constexpr auto Vector<T>::contains(T const & value) const noexcept
//    -> bool requires(!std::floating_point<T>)
//{
//  for (Size i = 0; i < _size; ++i) {
//    if (_data[i] == value) {
//      return true;
//    }
//  }
//  return false;
//}
//
//// A classic abs(a - b) <= epsilon comparison
// template <class T>
// requires(std::is_arithmetic_v<T> && !std::unsigned_integral<T>) PURE HOSTDEV
//     constexpr auto isApprox(Vector<T> const & a, Vector<T> const & b,
//                             T const epsilon) noexcept -> bool
//{
//   if (a.size() != b.size()) {
//     return false;
//   }
//   for (Size i = 0; i < a.size(); ++i) {
//     if (std::abs(a[i] - b[i]) > epsilon) {
//       return false;
//     }
//   }
//   return true;
// }
//
// template <class T>
// requires(std::unsigned_integral<T>) PURE HOSTDEV
//     constexpr auto isApprox(Vector<T> const & a, Vector<T> const & b,
//                             T const epsilon) noexcept -> bool
//{
//   if (a.size() != b.size()) {
//     return false;
//   }
//   for (Size i = 0; i < a.size(); ++i) {
//     T const diff = a[i] > b[i] ? a[i] - b[i] : b[i] - a[i];
//     if (diff > epsilon) {
//       return false;
//     }
//   };
//   return true;
// }

// ---------------------------------------------------------------------------
// Hidden methods
// ---------------------------------------------------------------------------
template <class T, class Allocator>
PURE HOSTDEV [[nodiscard]] constexpr HIDDEN auto
// cppcheck-suppress functionStatic
Vector<T, Allocator>::alloc() noexcept -> Allocator &
{
  return _end_cap.second;
}

template <class T, class Allocator>
PURE HOSTDEV [[nodiscard]] constexpr HIDDEN auto
Vector<T, Allocator>::alloc() const noexcept -> Allocator const &
{
  return _end_cap.second;
}

template <class T, class Allocator>
PURE HOSTDEV [[nodiscard]] constexpr HIDDEN auto
// cppcheck-suppress functionStatic
Vector<T, Allocator>::endcap() noexcept -> Ptr &
{
  return _end_cap.first;
}

template <class T, class Allocator>
PURE HOSTDEV [[nodiscard]] constexpr HIDDEN auto
Vector<T, Allocator>::endcap() const noexcept -> Ptr const &
{
  return _end_cap.first;
}

//  Default constructs n objects starting at _end
//  Precondition:  n > 0
//  Precondition:  size() + n <= capacity()
//  Postcondition:  size() == size() + n
template <class T, class Allocator>
HOSTDEV constexpr HIDDEN void
Vector<T, Allocator>::constructAtEnd(Size n)
{
  assert(n > 0);
  assert(size() + n <= capacity());
  ConstructTransaction tx(*this, n);
  ConstPtr new_end = tx.new_end;
  for (Ptr pos = tx.pos; pos != new_end; tx.pos = ++pos) {
    AllocTraits::construct(this->alloc(), pos);
  }
}

//template <class InputIterator, class Sentinel>
//HOSTDEV constexpr HIDDEN void
//constructAtEnd(InputIterator first, Sentinel last, Size n)
//{
////  ConstructTransaction tx(*this, __n);
////  tx.pos = uninitializedAllocatorCopy(alloc(), first, last, tx.pos);
//}

template <class T, class Allocator>
HOSTDEV constexpr HIDDEN void
Vector<T, Allocator>::destructAtEnd(Ptr new_last) noexcept
{
  Ptr soon_to_be_end = _end;
  while (new_last != soon_to_be_end) {
    AllocTraits::destroy(alloc(), --soon_to_be_end);
  }
  _end = new_last;
}

template <class T, class Allocator>
HOSTDEV constexpr HIDDEN void
Vector<T, Allocator>::clearMemory() noexcept
{
  destructAtEnd(_begin);
}

//  Allocate space for n objects
//  Precondition:  _begin == _end == endcap() == 0
//  Precondition:  n > 0
//  Postcondition:  capacity() >= n
//  Postcondition:  size() == 0
template <class T, class Allocator>
HOSTDEV constexpr HIDDEN void
Vector<T, Allocator>::allocate(Size const n)
{
  auto const allocation = AllocTraits::allocateAtLeast(alloc(), n);
  _begin = allocation.ptr;
  _end = allocation.ptr;
  endcap() = _begin + allocation.count;
}

template <class T, class Allocator>
template <class InputIterator, class Sentinel>
constexpr HIDDEN void
Vector<T, Allocator>::initWithSize(InputIterator first, Sentinel last, Size n)
{
  destroy_vector(*this)();
  if (n > 0) {
    allocate(n);
    constructAtEnd(first, last, n);
  }
}

} // namespace um2
