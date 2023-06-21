#pragma once

#include <um2/common/memory.hpp>

#include <cuda/std/bit> // cuda::std::bit_ceil
#include <cuda/std/utility> // cuda::std::pair

//#include <cmath>            // std::abs
//#include <cstring>          // memcpy
//#include <initializer_list> // std::initializer_list

namespace um2
{

// -----------------------------------------------------------------------------
// VECTOR
// -----------------------------------------------------------------------------
// An std::vector-like class.
//
// https://en.cppreference.com/w/cpp/container/vector

// The following std::vector functions still need to be implemented:
//    explicit vector(size_type n);
//    explicit vector(size_type n, const allocator_type&); // C++14
//    vector(size_type n, const value_type& value, const allocator_type& =
//    allocator_type()); template <class InputIterator>
//        vector(InputIterator first, InputIterator last, const allocator_type& =
//        allocator_type());
//    template<container-compatible-range<T> R>
//      constexpr vector(from_range_t, R&& rg, const Allocator& = Allocator()); // C++23
//    vector(const vector& x);
//    vector(vector&& x)
//        noexcept(is_nothrow_move_constructible<allocator_type>::value);
//    vector(initializer_list<value_type> il);
//    vector(initializer_list<value_type> il, const allocator_type& a);
//    vector& operator=(const vector& x);
//    vector& operator=(vector&& x)
//        noexcept(
//             allocator_type::propagate_on_container_move_assignment::value ||
//             allocator_type::is_always_equal::value); // C++17
//    vector& operator=(initializer_list<value_type> il);
//    template <class InputIterator>
//        void assign(InputIterator first, InputIterator last);
//    template<container-compatible-range<T> R>
//      constexpr void assign_range(R&& rg); // C++23
//    void assign(size_type n, const value_type& u);
//    void assign(initializer_list<value_type> il);
//
//    reverse_iterator       rbegin() noexcept;
//    const_reverse_iterator rbegin()  const noexcept;
//    reverse_iterator       rend() noexcept;
//    const_reverse_iterator rend()    const noexcept;
//
//    const_reverse_iterator crbegin() const noexcept;
//    const_reverse_iterator crend()   const noexcept;
//
//    void reserve(size_type n);
//    void shrink_to_fit() noexcept;
//
//    void push_back(const value_type& x);
//    void push_back(value_type&& x);
//    template <class... Args>
//        reference emplace_back(Args&&... args); // reference in C++17
//    template<container-compatible-range<T> R>
//      constexpr void append_range(R&& rg); // C++23
//    void pop_back();
//
//    template <class... Args> iterator emplace(const_iterator position, Args&&... args);
//    iterator insert(const_iterator position, const value_type& x);
//    iterator insert(const_iterator position, value_type&& x);
//    iterator insert(const_iterator position, size_type n, const value_type& x);
//    template <class InputIterator>
//        iterator insert(const_iterator position, InputIterator first, InputIterator
//        last);
//    template<container-compatible-range<T> R>
//      constexpr iterator insert_range(const_iterator position, R&& rg); // C++23
//    iterator insert(const_iterator position, initializer_list<value_type> il);
//
//    iterator erase(const_iterator position);
//    iterator erase(const_iterator first, const_iterator last);
//
//    void clear() noexcept;
//
//    void resize(size_type sz);
//    void resize(size_type sz, const value_type& c);
//
//    void swap(vector&)
//        noexcept(allocator_traits<allocator_type>::propagate_on_container_swap::value ||
//                 allocator_traits<allocator_type>::is_always_equal::value);  // C++17
//
//    bool __invariants() const;

template <typename T, typename Allocator = BasicAllocator<T>>
// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
struct Vector {

  using Ptr = T *;
  using EndCap = cuda::std::pair<Ptr, Allocator>;
  using AllocTraits = AllocatorTraits<Allocator>;

private:
  Ptr _begin = nullptr;
  Ptr _end = nullptr;
  EndCap _end_cap = EndCap(nullptr, Allocator());
  
  // ---------------------------------------------------------------------------
  // HIDDEN
  // ---------------------------------------------------------------------------
  [[nodiscard]] constexpr HIDDEN auto
  // cppcheck-suppress functionConst
  alloc() noexcept -> Allocator &
  {
    return _end_cap.second;
  }

  [[nodiscard]] constexpr HIDDEN auto
  alloc() const noexcept -> Allocator const &
  {
    return _end_cap.second;
  }

  [[nodiscard]] constexpr HIDDEN auto
  // cppcheck-suppress functionConst
  endcap() noexcept -> Allocator &
  {
    return _end_cap.first;
  }

  [[nodiscard]] constexpr HIDDEN auto
  endcap() const noexcept -> Allocator const &
  {
    return _end_cap.first;
  }

  constexpr HIDDEN void
  // NOLINTNEXTLINE(readability-identifier-naming)
  destruct_at_end(Ptr new_last) noexcept
  {
    Ptr soon_to_be_end = _end;
    while (new_last != soon_to_be_end) {
      AllocTraits::destroy(alloc(), --soon_to_be_end);
    }
    _end = new_last;
  }

  constexpr HIDDEN void
  // NOLINTNEXTLINE(readability-identifier-naming)
  clear_mem() noexcept
  {
    destruct_at_end(_begin);
  }

  //  Allocate space for n objects
  //  Precondition:  _begin == _end == endcap() == 0
  //  Precondition:  n > 0
  //  Postcondition:  capacity() >= n
  //  Postcondition:  size() == 0
  constexpr HIDDEN void
  allocate(Size const n)
  {
    auto const allocation = AllocTraits::allocate_at_least(alloc(), n);
    _begin = allocation.ptr;
    _end = allocation.ptr;
    endcap() = _begin + allocation.count;
  }

public:
  // Ignore UM2 naming convention to match std lib functions
  // NOLINTBEGIN(readability-identifier-naming)

  // -----------------------------------------------------------------------------
  // Constructors
  // -----------------------------------------------------------------------------

  constexpr Vector() noexcept(noexcept(Allocator())) = default;

  HOSTDEV constexpr explicit Vector(Allocator const & a) noexcept;

  HOSTDEV explicit Vector(Size n);
  ////
  ////  HOSTDEV Vector(Size n, T const & value);
  ////
  ////    HOSTDEV constexpr Vector(Vector const & v);
  ////
  ////  HOSTDEV Vector(Vector && v) noexcept;
  ////
  ////  // cppcheck-suppress noExplicitConstructor
  ////  // NOLINTNEXTLINE(google-explicit-constructor)
  ////  Vector(std::initializer_list<T> const & list);
  ////

  // -----------------------------------------------------------------------------
  // Destructor
  // -----------------------------------------------------------------------------
  HOSTDEV constexpr ~Vector() noexcept
  {
    if (_begin != nullptr) {
      clear_mem();
      AllocTraits::deallocate(alloc(), _begin, capacity());
    }
  }

  // -----------------------------------------------------------------------------
  // Accessors
  // -----------------------------------------------------------------------------

  PURE HOSTDEV [[nodiscard]] constexpr auto
  get_allocator() const noexcept -> Allocator;

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
  
  HOSTDEV constexpr void reserve(Size n);

  ////  HOSTDEV void clear() noexcept;
  ////
  ////
  ////  HOSTDEV void resize(Size n);
  ////
  ////  HOSTDEV inline void push_back(T const & value);
  ////
  ////  PURE HOSTDEV [[nodiscard]] constexpr auto empty() const -> bool;
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

  NDEBUG_PURE HOSTDEV constexpr auto
  // cppcheck-suppress functionConst
  operator[](Size i) noexcept -> T &;

  NDEBUG_PURE HOSTDEV constexpr auto
  operator[](Size i) const noexcept -> T const &;
  ////
  ////  HOSTDEV auto operator=(Vector const & v) -> Vector &;
  ////
  ////  HOSTDEV auto operator=(Vector && v) noexcept -> Vector &;
  ////
  ////  PURE HOSTDEV constexpr auto operator==(Vector const & v) const noexcept -> bool;
  //
  //  // NOLINTEND(readability-identifier-naming)
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
template <typename Allocator>
struct Vector<bool, Allocator> {
};

} // namespace um2

#include "Vector.inl"
