namespace um2
{

//==============================================================================
// Hidden
//==============================================================================

template <class T>
HOSTDEV constexpr void
Vector<T>::allocate(Size n) noexcept
{
  assert(n < max_size());
  assert(_begin == nullptr);
  _begin = static_cast<T *>(::operator new(static_cast<size_t>(n) * sizeof(T)));
  _end = _begin;
  _end_cap = _begin + n;
}

template <class T>
HOSTDEV constexpr void
Vector<T>::construct_at_end(Size n) noexcept
{
  Ptr new_end = _end + n;
  for (Ptr pos = _end; pos != new_end; ++pos) {
    construct_at(pos);
  }
  _end = new_end;
}

template <class T>
HOSTDEV constexpr void
Vector<T>::construct_at_end(Size n, T const & value) noexcept
{
  Ptr new_end = _end + n;
  for (Ptr pos = _end; pos != new_end; ++pos) {
    construct_at(pos, value);
  }
  _end = new_end;
}

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
PURE HOSTDEV constexpr auto
Vector<T>::recommend(Size new_size) const noexcept -> Size
{
  return um2::max(2 * capacity(), new_size);
}

template <class T>
HOSTDEV constexpr void
Vector<T>::grow(Size n) noexcept
{
  Size const current_size = size();
  Size const new_size = current_size + n;
  Size const new_capacity = recommend(new_size);
  Ptr new_begin =
      static_cast<T *>(::operator new(static_cast<size_t>(new_capacity) * sizeof(T)));
  Ptr new_end = new_begin;
  // Move the elements over
  for (Ptr old_pos = _begin; old_pos != _end; ++old_pos, ++new_end) {
    um2::construct_at(new_end, um2::move(*old_pos));
  }
  // Destroy the old elements
  destruct_at_end(_begin);
  // Update the pointers
  _begin = new_begin;
  _end = new_end;
  _end_cap = _begin + new_capacity;
}

template <class T>
HOSTDEV constexpr void
Vector<T>::append_default(Size n) noexcept
{
  // If we have enough capacity, just construct the new elements
  // Otherwise, allocate a new buffer and move the elements over
  if (static_cast<Size>(_end_cap - _end) < n) {
    grow(n);
  }
  // Construct the new elements
  construct_at_end(n);
}

//==============================================================================-
// Constructors
//==============================================================================-

template <class T>
HOSTDEV constexpr Vector<T>::Vector(Size const n) noexcept
{
  allocate(n);
  construct_at_end(n);
}

template <class T>
HOSTDEV constexpr Vector<T>::Vector(Size const n, T const & value) noexcept
{
  allocate(n);
  construct_at_end(n, value);
}

template <class T>
HOSTDEV constexpr Vector<T>::Vector(Vector<T> const & v) noexcept
{
  Size const n = v.size();
  allocate(n);
  construct_at_end(n);
  copy(v._begin, v._end, _begin);
}

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
HOSTDEV constexpr Vector<T>::Vector(std::initializer_list<T> const & list) noexcept
{
  // Initializer lists can't be moved from, so we have to copy. Pretty silly.
  Size const n = static_cast<Size>(list.size());
  allocate(n);
  construct_at_end(n);
  copy(list.begin(), list.end(), _begin);
}

//==============================================================================-
// Destructor
//==============================================================================-

template <class T>
HOSTDEV constexpr Vector<T>::~Vector() noexcept
{
  if (_begin != nullptr) {
    this->destruct_at_end(_begin);
    ::operator delete(_begin);
  }
}

//==============================================================================-
// Accessors
//==============================================================================-

template <class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T>::max_size() noexcept -> Size
{
  return sizeMax() / sizeof(T);
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
Vector<T>::size() const noexcept -> Size
{
  return static_cast<Size>(_end - _begin);
}

template <class T>
PURE HOSTDEV constexpr auto
Vector<T>::capacity() const noexcept -> Size
{
  return static_cast<Size>(_end_cap - _begin);
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
  return begin();
}

template <class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T>::cend() const noexcept -> T const *
{
  return end();
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
  assert(size() > 0);
  return *(_end - 1);
}

template <class T>
PURE HOSTDEV [[nodiscard]] constexpr auto
Vector<T>::back() const noexcept -> T const &
{
  assert(size() > 0);
  return *(_end - 1);
}

template <class T>
// cppcheck-suppress functionConst justification: can't be const
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
PURE HOSTDEV constexpr auto
Vector<T>::operator[](Size const i) noexcept -> T &
{
  assert(i < size());
  return _begin[i];
}

template <class T>
PURE HOSTDEV constexpr auto
Vector<T>::operator[](Size const i) const noexcept -> T const &
{
  assert(i < size());
  return _begin[i];
}

template <class T>
HOSTDEV constexpr auto
Vector<T>::operator=(Vector<T> const & v) noexcept -> Vector<T> &
{
  if (this != addressof(v)) {
    destruct_at_end(_begin);
    ::operator delete(_begin);
    _begin = nullptr;
    allocate(v.size());
    construct_at_end(v.size());
    copy(v.begin(), v.end(), _begin);
  }
  return *this;
}

template <class T>
HOSTDEV constexpr auto
Vector<T>::operator=(Vector<T> && v) noexcept -> Vector<T> &
{
  if (this != addressof(v)) {
    destruct_at_end(_begin);
    ::operator delete(_begin);
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
HOSTDEV constexpr auto
Vector<T>::operator=(std::initializer_list<T> const & list) noexcept -> Vector &
{
  destruct_at_end(_begin);
  ::operator delete(_begin);
  _begin = nullptr;
  allocate(static_cast<Size>(list.size()));
  construct_at_end(static_cast<Size>(list.size()));
  copy(list.begin(), list.end(), _begin);
  return *this;
}

template <class T>
PURE constexpr auto
Vector<T>::operator==(Vector<T> const & v) const noexcept -> bool
{
  return size() == v.size() && std::equal(begin(), end(), v.begin());
}

//==============================================================================
// Methods
//==============================================================================

template <class T>
HOSTDEV constexpr void
Vector<T>::clear() noexcept
{
  um2::destroy(_begin, _end);
  _end = _begin;
}

template <class T>
HOSTDEV constexpr void
Vector<T>::resize(Size const n) noexcept
{
  Size const cs = size();
  // If we are shrinking, destroy the elements that are no longer needed
  // If we are growing, default construct the new elements
  if (cs < n) {
    append_default(n - cs);
  } else if (cs > n) {
    destruct_at_end(_begin + n);
  }
}

template <class T>
HOSTDEV constexpr void
Vector<T>::push_back(T const & value) noexcept
{
  if (_end == _end_cap) {
    grow(1);
  }
  construct_at(_end, value);
  ++_end;
}

template <class T>
HOSTDEV constexpr void
Vector<T>::push_back(T && value) noexcept
{
  if (_end == _end_cap) {
    this->grow(1);
  }
  um2::construct_at(_end, um2::move(value));
  ++_end;
}

template <class T>
HOSTDEV constexpr void
Vector<T>::push_back(Size const n, T const & value) noexcept
{
  // If we have enough capacity, just construct the new elements
  // Otherwise, allocate a new buffer and move the elements over
  if (static_cast<Size>(_end_cap - _end) < n) {
    this->grow(n);
  }
  // Construct the new elements
  construct_at_end(n, value);
}

template <typename T>
constexpr void
sortPermutation(Vector<T> const & v, Vector<Size> & perm) noexcept
{
  perm.resize(v.size());
  std::iota(perm.begin(), perm.end(), 0);
  std::sort(perm.begin(), perm.end(),
            [&v](Size const i, Size const j) { return v[i] < v[j]; });
}

template <typename T>
constexpr void
applyPermutation(Vector<T> & v, Vector<Size> const & perm) noexcept
{
  // Verify that perm is a permutation
  // (contains all elements of [0, v.size()) exactly once)
#ifndef NDEBUG
  Vector<int8_t> seen(v.size(), 0);
  for (Size const i : perm) {
    assert(i < v.size());
    assert(seen[i] == 0);
    seen[i] = 1;
  }
  assert(std::count(seen.cbegin(), seen.cend(), 1) == v.size());
#endif
  // Apply the permutation in-place by iterating over cycles.
  Vector<int8_t> done(v.size());
  for (Size i = 0; i < v.size(); ++i) {
    if (done[i] == 1) {
      continue;
    }
    done[i] = true;
    Size prev_j = i;
    Size j = perm[i];
    while (i != j) {
      std::swap(v[prev_j], v[j]);
      done[j] = true;
      prev_j = j;
      j = perm[j];
    }
  }
}

} // namespace um2
