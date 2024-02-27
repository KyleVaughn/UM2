#pragma once

#include <um2/config.hpp>

//#include <um2/stdlib/assert.hpp>
//#include <um2/stdlib/algorithm/copy.hpp>
////#include <um2/stdlib/math.hpp>      // min
//#include <um2/stdlib/memory/addressof.hpp> // addressof
//#include <um2/stdlib/utility/move.hpp>     // move
////
////#include <cstring> // memcpy, strcmp
////#include <string>  // std::string
//
//// The length of a string, not including the null terminator.
//inline constexpr auto
//length(char const * s) -> Int
//{
//  Int n = 0;
//  while (s[n] != '\0') {
//    ++n;
//  }
//  return n;
//}

//==============================================================================
// STRING VIEW
//==============================================================================

namespace um2
{

class StringView
{
  public:
  using Ptr = char *;
  using ConstPtr = char const *;

  private:
  ConstPtr _data;
  Int _size;

  //==============================================================================
  // Private member functions
  //==============================================================================

public:

  //==============================================================================
  // Constructors
  //==============================================================================

  HOSTDEV constexpr StringView() noexcept;
//
//  HOSTDEV constexpr StringView(StringView const & s) noexcept;
//
//  HOSTDEV constexpr StringView(StringView && s) noexcept;
//
////  template <uint64_t N>
////  HOSTDEV constexpr StringView(char const (&s)[N]) noexcept;
//
//  // NOLINTBEGIN(google-explicit-constructor) match std::string
//  HOSTDEV constexpr StringView(char const * s) noexcept;
//  // NOLINTEND(google-explicit-constructor)
////
////  HOSTDEV constexpr StringView(char const * begin, char const * end) noexcept;
////
////  HOSTDEV constexpr StringView(char const * s, Int n) noexcept;
////
////  template <std::integral T>
////  explicit constexpr StringView(T x) noexcept;
////
////  template <std::floating_point T>
////  explicit constexpr StringView(T x) noexcept;
////
//  //==============================================================================
//  // Destructor
//  //==============================================================================
//
//  HOSTDEV inline constexpr ~StringView() noexcept;
//
  //==============================================================================
  // Accessors
  //==============================================================================

//  // The number of characters that can be held without reallocating storage.
//  // Does not include the null terminator.
//  PURE HOSTDEV [[nodiscard]] constexpr auto
//  capacity() const noexcept -> Int;
//
//  PURE HOSTDEV [[nodiscard]] constexpr auto
//  data() noexcept -> Ptr;
//
//  PURE HOSTDEV [[nodiscard]] constexpr auto
//  data() const noexcept -> ConstPtr;
//
//  PURE HOSTDEV [[nodiscard]] constexpr auto
//  isLong() const noexcept -> bool;
//
//  // Not including the null terminator.
//  PURE HOSTDEV [[nodiscard]] constexpr auto
//  size() const noexcept -> Int;
//
//  //==============================================================================
//  // Operators
//  //==============================================================================
//
////  HOSTDEV constexpr auto
////  operator=(StringView const & s) noexcept -> StringView &;
//
//  HOSTDEV constexpr auto
//  operator=(StringView && s) noexcept -> StringView &;
//
////  constexpr auto
////  operator=(std::string const & s) noexcept -> StringView &;
////
////  constexpr auto
////  operator=(std::string && s) noexcept -> StringView &;
////
////  template <uint64_t N>
////  HOSTDEV constexpr auto
////  operator=(char const (&s)[N]) noexcept -> StringView &;
////
////  PURE HOSTDEV constexpr auto
////  operator==(StringView const & s) const noexcept -> bool;
////
////  PURE HOSTDEV constexpr auto
////  operator!=(StringView const & s) const noexcept -> bool;
////
////  PURE HOSTDEV constexpr auto
////  operator<(StringView const & s) const noexcept -> bool;
////
////  PURE HOSTDEV constexpr auto
////  operator<=(StringView const & s) const noexcept -> bool;
////
////  PURE HOSTDEV constexpr auto
////  operator>(StringView const & s) const noexcept -> bool;
////
////  PURE HOSTDEV constexpr auto
////  operator>=(StringView const & s) const noexcept -> bool;
////
////  PURE HOSTDEV constexpr auto
////  operator[](Int i) noexcept -> char &;
////
////  PURE HOSTDEV constexpr auto
////  operator[](Int i) const noexcept -> char const &;
////
////  HOSTDEV constexpr auto
////  operator+=(StringView const & s) noexcept -> StringView &;
////
////  HOSTDEV constexpr auto
////  operator+=(char c) noexcept -> StringView &;
////
////  //==============================================================================
////  // Methods
////  //==============================================================================
////
////  PURE HOSTDEV [[nodiscard]] constexpr auto
////  compare(StringView const & s) const noexcept -> int;
////
////  PURE HOSTDEV [[nodiscard]] constexpr auto
////  c_str() const noexcept -> char const *;
////
////  PURE HOSTDEV [[nodiscard]] constexpr auto
////  ends_with(StringView const & s) const noexcept -> bool;
////
////  template <uint64_t N>
////  PURE HOSTDEV [[nodiscard]] constexpr auto
////  ends_with(char const (&s)[N]) const noexcept -> bool;
////
////  PURE HOSTDEV [[nodiscard]] constexpr auto
////  find_last_of(char c) const noexcept -> Int;
////
//  PURE HOSTDEV [[nodiscard]] constexpr auto
//  starts_with(ConstPtr s) const noexcept -> bool;
////
////  PURE HOSTDEV [[nodiscard]] constexpr auto
////  substr(Int pos, Int len = npos) const -> StringView;
////
}; // class StringView

//////==============================================================================
////// Non-member operators
//////==============================================================================
////
////HOSTDEV constexpr auto
////operator+(StringView l, StringView const & r) noexcept -> StringView;
////
////template <uint64_t N>
////HOSTDEV constexpr auto
////operator+(StringView l, char const (&r)[N]) noexcept -> StringView;
////
////template <uint64_t N>
////HOSTDEV constexpr auto
////operator+(char const (&l)[N], StringView const & r) noexcept -> StringView;
////
//////==============================================================================
////// Non-member functions
//////==============================================================================
////
////template <typename T>
////constexpr auto
////toStringView(T const & t) noexcept -> StringView;
////
//////==============================================================================
////// MACROS
//////==============================================================================
////// To maintain constexpr-ness and readability, we define a few macros to avoid
////// repeating ourselves
//////
////// gcc seems to have a bug that causes it to generate a call to memmove that
////// is out of bounds when using std::copy with -O3. Therefore, we write out a basic
////// copy loop instead.
////
///*
//
////#define COPY_LOOP(first, last, dest)                                                      \
////  auto f = (first);                                                                       \
////  auto l = (last);                                                                        \
////  auto d = (dest);                                                                        \
////  while (f != l) {                                                                        \
////    *d = *f;                                                                              \
////    ++f;                                                                                  \
////    ++d;                                                                                  \
////  }
////
////// mimic allocateAndCopy(char const *, uint64_t)
////#define LONG_ALLOCATE_AND_COPY(ptr, num_elem)                                            \
////  char const * ss = (ptr);                                                               \
////  uint64_t const nn = (num_elem);                                                        \
////  ASSERT_ASSUME(nn > 0);                                                                 \
////  _r.l.is_long = 1;                                                                      \
////  _r.l.cap = (nn - 1) & long_cap_mask;                                                   \
////  _r.l.size = nn - 1;                                                                    \
////  _r.l.data = static_cast<char *>(::operator new(nn));                                   \
////  COPY_LOOP(ss, ss + nn, _r.l.data);
////
////#define SHORT_COPY(ptr, num_elem)                                                        \
////  char const * ss = (ptr);                                                               \
////  uint64_t const nn = (num_elem);                                                        \
////  ASSERT_ASSUME(nn > 0);                                                                 \
////  _r.s.is_long = 0;                                                                      \
////  _r.s.size = (nn - 1) & short_size_mask;                                                \
////  COPY_LOOP(ss, ss + nn, _r.s.data);
////
//*/
////==============================================================================
//// Private functions
////==============================================================================
//
//HOSTDEV inline constexpr StringView::~StringView() noexcept
//{
//  if (isLong()) {
//    ::operator delete(getLongPointer());
//  }
//}
//
//// Retains the representation of the string, but sets the size to 0.
//HOSTDEV inline constexpr void
//StringView::clear() noexcept
//{
//  if (isLong()) {
//    *getLongPointer() = '\0';
//    setLongSize(0);
//  } else {
//    *getShortPointer() = '\0';
//    setShortSize(0);
//  }
//}
//
//HOSTDEV inline constexpr void
//StringView::clearAndShrink() noexcept
//{
//  clear();
//  if (isLong()) {
//    ::operator delete(getLongPointer());
//    _r = Rep();
//  }
//}
//
////HOSTDEV constexpr void
////StringView::copyAssignAlloc(StringView const & s) noexcept
////{
////  if (!s.isLong()) {
//////    // If s is a short string, we can just copy the whole struct after
//////    // clearing the current string.
//////    clearAndShrink();
//////    _r = s._r;
//////  } else {
//////    // If s is a long string, we need to allocate new memory.
//////    Ptr alloc = static_cast<Ptr>(::operator new(s.getLongCap()));
//////    if (isLong()) {
//////      ::operator delete(getLongPointer());
//////    }
//////    setLongPointer(alloc);
//////    setLongCap(s.getLongCap());
//////    setLongSize(s.getLongSize());
////  }
////}
//
//PURE HOSTDEV constexpr auto
//StringView::getLongSize() const noexcept -> uint64_t
//{
//  return _r.l.size;
//}
//
//PURE HOSTDEV constexpr auto
//StringView::getShortSize() const noexcept -> uint64_t
//{
//  return _r.s.size;
//}
//
//PURE HOSTDEV constexpr auto
//StringView::getLongCap() const noexcept -> uint64_t
//{
//  return _r.l.cap;
//}
//
//PURE HOSTDEV constexpr auto
//// NOLINTNEXTLINE(readability-make-member-function-const) we offer const next
//StringView::getLongPointer() noexcept -> Ptr
//{
//  return _r.l.data;
//}
//
//PURE HOSTDEV constexpr auto
//StringView::getLongPointer() const noexcept -> ConstPtr
//{
//  return _r.l.data;
//}
//
//PURE HOSTDEV constexpr auto
//StringView::getShortPointer() noexcept -> Ptr
//{
//  return um2::addressof(_r.s.data[0]);
//}
//
//PURE HOSTDEV constexpr auto
//StringView::getShortPointer() const noexcept -> ConstPtr
//{
//  return um2::addressof(_r.s.data[0]);
//}
//
//PURE HOSTDEV constexpr auto
//StringView::getPointer() noexcept -> Ptr
//{
//  return isLong() ? getLongPointer() : getShortPointer();
//}
//
//PURE HOSTDEV constexpr auto
//StringView::getPointer() const noexcept -> ConstPtr
//{
//  return isLong() ? getLongPointer() : getShortPointer();
//}
//
//// n does not include the null terminator
//CONST HOSTDEV constexpr auto
//StringView::fitsInShort(uint64_t n) noexcept -> bool
//{
//  return n < min_cap;
//}
//
//HOSTDEV constexpr void
//StringView::init(ConstPtr s, uint64_t size) noexcept
//{
//  ASSERT(s != nullptr);
//  Ptr p = nullptr;
//  if (fitsInShort(size)) {
//    setShortSize(size);
//    p = getShortPointer();
//  } else {
//    p = static_cast<Ptr>(::operator new(size + 1));
//    setLongPointer(p);
//    setLongCap(size + 1);
//    setLongSize(size);
//  }
//  um2::copy(s, s + size, p);
//  p[size] = '\0';
//}
//
//HOSTDEV constexpr void
//StringView::setLongCap(uint64_t cap) noexcept
//{
//  _r.l.cap = cap & long_cap_mask;
//  _r.l.is_long = true;
//}
//
//HOSTDEV constexpr void
//StringView::setLongPointer(Ptr p) noexcept
//{
//  _r.l.data = p;
//}
//
//HOSTDEV constexpr void
//StringView::setLongSize(uint64_t size) noexcept
//{
//  _r.l.size = size;
//}
//
//HOSTDEV constexpr void
//StringView::setShortSize(uint64_t size) noexcept
//{
//  ASSERT(size < min_cap);
//  _r.s.size = size & short_size_mask;
//  _r.s.is_long = false;
//}
//
////template <typename T>
////constexpr auto
////toStringView(T const & t) noexcept -> StringView
////{
////  return StringView(t);
////}
////
////HOSTDEV constexpr auto
////operator+(StringView l, StringView const & r) noexcept -> StringView
////{
////  l += r;
////  return l;
////}
////
////template <uint64_t N>
////HOSTDEV constexpr auto
////operator+(StringView l, char const (&r)[N]) noexcept -> StringView
////{
////  l += StringView(r);
////  return l;
////}
////
////template <uint64_t N>
////HOSTDEV constexpr auto
////operator+(char const (&l)[N], StringView const & r) noexcept -> StringView
////{
////  StringView tmp(l);
////  tmp += r;
////  return tmp;
////}
////
//==============================================================================
// Constructors
//==============================================================================

HOSTDEV constexpr StringView::StringView() noexcept
    : _data(nullptr), _size(0)
{
}

//HOSTDEV constexpr StringView::StringView(StringView const & s) noexcept
//{
//  if (!s.isLong()) {
//    // If this is a short string, it is trivially copyable
//    _r = s._r;
//  } else {
//    init(s.getLongPointer(), s.getLongSize());
//  }
//}
//
//HOSTDEV constexpr StringView::StringView(StringView && s) noexcept
//    : _r(um2::move(s._r))
//{
//  // If short string, we can copy trivially
//  // If long string, we need to move the data.
//  // Since the data is a pointer, we can just copy the pointer.
//  // Therefore, either way, we can just copy the whole struct.
//  s._r = Rep();
//}
//
//////template <uint64_t N>
//////HOSTDEV constexpr StringView::StringView(char const (&s)[N]) noexcept
//////{
//////  // N includes the null terminator
//////  char const * p = addressof(s[0]);
//////  if constexpr (N <= min_cap) {
//////    SHORT_COPY(p, N);
//////    ASSERT(_r.s.data[N - 1] == '\0');
//////  } else {
//////    LONG_ALLOCATE_AND_COPY(p, N);
//////    ASSERT(_r.l.data[N - 1] == '\0');
//////  }
//////}
//
//HOSTDEV constexpr StringView::StringView(char const * s) noexcept
//{
//  ASSERT(s != nullptr);
//  // Get the length of the string (not including null terminator)
//  auto const n = static_cast<uint64_t>(length(s));
//  ASSERT(n > 0);
//  init(s, n);
//}
//
//////HOSTDEV constexpr StringView::StringView(char const * s, Int const n) noexcept
//////{
//////  // Short string
//////  auto const cap = static_cast<uint64_t>(n);
//////  if (cap + 1 <= min_cap) {
//////    SHORT_COPY(s, cap + 1);
//////    _r.s.data[cap] = '\0';
//////  } else {
//////    LONG_ALLOCATE_AND_COPY(s, cap + 1);
//////    _r.l.data[cap] = '\0';
//////  }
//////}
//////
//////HOSTDEV constexpr StringView::StringView(char const * begin, char const * end) noexcept
//////{
//////  // begin and end are pointers to the first and one past the last character
//////  // of the string, respectively.
//////  //
//////  // "test" -> begin = &t, end = &t + 4
//////  // Hence, end is not a valid memory location, nor necessarily the null
//////  // terminator.
//////  auto const n = static_cast<uint64_t>(end - begin);
//////  ASSERT(n > 0);
//////  if (n + 1 <= min_cap) {
//////    _r.s.is_long = 0;
//////    _r.s.size = n & short_size_mask;
//////    auto * dest = addressof(_r.s.data[0]); 
//////    while (begin != end) {              
//////      *dest = *begin;                  
//////      ++begin;                        
//////      ++dest;                        
//////    }
//////    *dest = '\0';
//////  } else {
//////    _r.l.is_long = 1;                
//////    _r.l.cap = n & long_cap_mask;   
//////    _r.l.size = n;                 
//////    _r.l.data = static_cast<char *>(::operator new(n));
//////    auto * dest = _r.l.data; 
//////    while (begin != end) {              
//////      *dest = *begin;                  
//////      ++begin;                        
//////      ++dest;                        
//////    }
//////    *dest = '\0';
//////  }
//////}
//////
//////// std::to_string should not allocate here due to small string optimization, so
//////// we are fine keeping these for now. They should be replaced with something
//////// like snprintf in the future.
//////template <std::integral T>
//////constexpr StringView::StringView(T x) noexcept
//////{
//////  // A 64-bit integer can have at most 20 chars
//////  std::string const s = std::to_string(x);
//////  auto const cap = s.size();
//////  ASSERT_ASSUME(cap < min_cap);
//////  SHORT_COPY(s.data(), cap + 1);
//////  _r.s.data[cap] = '\0';
//////}
//////
//////template <std::floating_point T>
//////constexpr StringView::StringView(T x) noexcept
//////{
//////  // A 64-bit floating point number can have at most 24 chars, but
//////  // many numbers will have fewer than 24 chars. So, we ASSERT that
//////  // this is the case.
//////  std::string const s = std::to_string(x);
//////  auto const cap = s.size();
//////  ASSERT_ASSUME(cap < min_cap);
//////  SHORT_COPY(s.data(), cap + 1);
//////  _r.s.data[cap] = '\0';
//////}
//////
////==============================================================================
//// Accessors
////==============================================================================
//
//PURE HOSTDEV constexpr auto
//StringView::isLong() const noexcept -> bool
//{
//  return _r.s.is_long;
//}
//
//PURE HOSTDEV constexpr auto
//StringView::size() const noexcept -> Int
//{
//  return isLong() ? static_cast<Int>(getLongSize()) : static_cast<Int>(getShortSize());
//}
//
//// Allocated bytes - 1 for null terminator
//PURE HOSTDEV constexpr auto
//StringView::capacity() const noexcept -> Int
//{
//  return isLong() ? static_cast<Int>(getLongCap()) - 1 : static_cast<Int>(min_cap) - 1;
//}
//
//PURE HOSTDEV constexpr auto
//StringView::data() noexcept -> Ptr
//{
//  return getPointer();
//}
//
//PURE HOSTDEV constexpr auto
//StringView::data() const noexcept -> ConstPtr
//{
//  return getPointer();
//}
//
////==============================================================================
//// Operators
////==============================================================================
//
////HOSTDEV constexpr auto
////StringView::operator=(StringView const & s) noexcept -> StringView &
////{
////  if (this != um2::addressof(s)) {
////    copyAssignAlloc(s);
////    // if s is a short string, we are done.
////    // Otherwise, we must now copy the data.
////    if (isLong()) {
////      um2::copy(s.getLongPointer(), s.getLongPointer() + s.getLongSize() + 1, getLongPointer());
////    }
////  }
////
////  return *this;
////}
//
//HOSTDEV constexpr auto
//StringView::operator=(StringView && s) noexcept -> StringView &
//{
//  if (this != addressof(s)) {
//    if (isLong()) {
//      ::operator delete(getLongPointer());
//    }
//    _r = s._r;
//    // We just copied the struct, so we are safe to overwrite s.
//    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
//    s._r = Rep();
//  }
//  return *this;
//}
//
//////// These std::string assignment operators are a bit inefficient, but the number of
//////// heap allocations is the same as if we had just copied the string, so it's not
//////// too bad.
//////constexpr auto
//////StringView::operator=(std::string const & s) noexcept -> StringView &
//////{
//////  StringView tmp(s.c_str());
//////  return *this = um2::move(tmp);
//////}
//////
//////constexpr auto
//////StringView::operator=(std::string && s) noexcept -> StringView &
//////{
//////  StringView tmp(s.c_str());
//////  return *this = um2::move(tmp);
//////}
//////
//////template <uint64_t N>
//////HOSTDEV constexpr auto
//////StringView::operator=(char const (&s)[N]) noexcept -> StringView &
//////{
//////  if (isLong()) {
//////    ::operator delete(_r.l.data);
//////  }
//////  // Short string
//////  if constexpr (N <= min_cap) {
//////    SHORT_COPY(s, N);
//////    ASSERT(_r.s.data[N - 1] == '\0');
//////  } else {
//////    LONG_ALLOCATE_AND_COPY(s, N);
//////    ASSERT(_r.l.data[N - 1] == '\0');
//////  }
//////  return *this;
//////}
//////
//////PURE HOSTDEV constexpr auto
//////StringView::operator==(StringView const & s) const noexcept -> bool
//////{
//////  Int const l_size = size();
//////  Int const r_size = s.size();
//////  if (l_size != r_size) {
//////    return false;
//////  }
//////  char const * l_data = data();
//////  char const * r_data = s.data();
//////  for (Int i = 0; i < l_size; ++i) {
//////    // NOLINTNEXTLINE
//////    if (*l_data != *r_data) {
//////      return false;
//////    }
//////    ++l_data;
//////    ++r_data;
//////  }
//////  return true;
//////}
//////
//////PURE HOSTDEV constexpr auto
//////StringView::operator!=(StringView const & s) const noexcept -> bool
//////{
//////  return !(*this == s);
//////}
//////
//////PURE HOSTDEV constexpr auto
//////StringView::operator<(StringView const & s) const noexcept -> bool
//////{
//////  return compare(s) < 0;
//////}
//////
//////PURE HOSTDEV constexpr auto
//////StringView::operator<=(StringView const & s) const noexcept -> bool
//////{
//////  return compare(s) <= 0;
//////}
//////
//////PURE HOSTDEV constexpr auto
//////StringView::operator>(StringView const & s) const noexcept -> bool
//////{
//////  return compare(s) > 0;
//////}
//////
//////PURE HOSTDEV constexpr auto
//////StringView::operator>=(StringView const & s) const noexcept -> bool
//////{
//////  return compare(s) >= 0;
//////}
//////
//////PURE HOSTDEV constexpr auto
//////StringView::operator[](Int i) noexcept -> char &
//////{
//////  return data()[i];
//////}
//////
//////PURE HOSTDEV constexpr auto
//////StringView::operator[](Int i) const noexcept -> char const &
//////{
//////  return data()[i];
//////}
//////
//////HOSTDEV constexpr auto
//////StringView::operator+=(StringView const & s) noexcept -> StringView &
//////{
//////  auto const new_size = static_cast<uint64_t>(size() + s.size());
//////  if (fitsInShort(new_size + 1)) {
//////    // This must be a short string, so we can just copy the data.
//////    ASSERT(!isLong());
//////    ASSERT(!s.isLong());
//////    ASSERT(new_size < min_cap);
//////    memcpy(getPointer() + size(), s.data(), static_cast<uint64_t>(s.size() + 1));
//////    _r.s.size = new_size & short_size_mask;
//////  } else {
//////    // Otherwise, we need to allocate a new string and copy the data.
//////    char * tmp = static_cast<char *>(::operator new(new_size + 1));
//////    memcpy(tmp, data(), static_cast<uint64_t>(size()));
//////    memcpy(tmp + size(), s.data(), static_cast<uint64_t>(s.size() + 1));
//////    if (isLong()) {
//////      ::operator delete(_r.l.data);
//////    }
//////    _r.l.is_long = 1;
//////    _r.l.cap = (new_size + 1) & long_cap_mask;
//////    _r.l.size = new_size;
//////    _r.l.data = tmp;
//////  }
//////  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks) Valgrind says this is fine
//////  return *this;
//////}
//////
//////HOSTDEV constexpr auto
//////StringView::operator+=(char const c) noexcept -> StringView &
//////{
//////  // If this is a short string and the size of the new string is less than
//////  // the capacity of the short string, we can just append the new string.
//////  auto const new_size = static_cast<uint64_t>(size() + 1);
//////  if (fitsInShort(new_size + 1)) {
//////    ASSERT(!isLong());
//////    _r.s.data[size()] = c;
//////    _r.s.data[size() + 1] = '\0';
//////    _r.s.size += 1;
//////  } else {
//////    // Otherwise, we need to allocate a new string and copy the data.
//////    char * tmp = static_cast<char *>(::operator new(new_size + 1));
//////    memcpy(tmp, data(), static_cast<uint64_t>(size()));
//////    tmp[size()] = c;
//////    tmp[size() + 1] = '\0';
//////    if (isLong()) {
//////      ::operator delete(_r.l.data);
//////    }
//////    _r.l.is_long = 1;
//////    _r.l.cap = (new_size + 1) & long_cap_mask;
//////    _r.l.size = new_size;
//////    _r.l.data = tmp;
//////  }
//////  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks) Valgrind says this is fine
//////  return *this;
//////}
//////
////==============================================================================
//// Member functions 
////==============================================================================
//
//////PURE HOSTDEV constexpr auto
//////StringView::compare(StringView const & s) const noexcept -> int
//////{
//////  Int const l_size = size();
//////  Int const r_size = s.size();
//////  Int const min_size = um2::min(l_size, r_size);
//////  char const * l_data = data();
//////  char const * r_data = s.data();
//////  for (Int i = 0; i < min_size; ++i) {
//////    if (*l_data != *r_data) {
//////      return static_cast<int>(*l_data) - static_cast<int>(*r_data);
//////    }
//////    ++l_data;
//////    ++r_data;
//////  }
//////  return static_cast<int>(l_size) - static_cast<int>(r_size);
//////}
//////
//////PURE HOSTDEV constexpr auto
//////StringView::c_str() const noexcept -> char const *
//////{
//////  return data();
//////}
//////
//PURE HOSTDEV constexpr auto
//StringView::starts_with(ConstPtr s) const noexcept -> bool
//{
//  if (size() < s.size()) {
//    return false;
//  }
//  char const * l_data = data();
//  char const * r_data = s.data();
//  for (Int i = 0; i < s.size(); ++i) {
//    if (*l_data != *r_data) {
//      return false;
//    }
//    ++l_data;
//    ++r_data;
//  }
//  return true;
//}
//
//////PURE HOSTDEV constexpr auto
//////StringView::ends_with(StringView const & s) const noexcept -> bool
//////{
//////  Int const l_size = size();
//////  Int const r_size = s.size();
//////  if (l_size < r_size) {
//////    return false;
//////  }
//////  char const * l_data = data() + l_size - r_size;
//////  char const * r_data = s.data();
//////  for (Int i = 0; i < r_size; ++i) {
//////    if (*l_data != *r_data) {
//////      return false;
//////    }
//////    ++l_data;
//////    ++r_data;
//////  }
//////  return true;
//////}
//////
//////template <uint64_t N>
//////PURE HOSTDEV constexpr auto
//////// NOLINTNEXTLINE(readability-identifier-naming) match std::string
//////StringView::ends_with(char const (&s)[N]) const noexcept -> bool
//////{
//////  return ends_with(StringView(s));
//////}
//////
//////PURE HOSTDEV constexpr auto
//////StringView::substr(Int pos, Int len) const -> StringView
//////{
//////  ASSERT(pos <= size());
//////  if (len == npos || pos + len > size()) {
//////    len = size() - pos;
//////  }
//////  return StringView{data() + pos, len};
//////}
//////
//////PURE HOSTDEV constexpr auto
//////StringView::find_last_of(char const c) const noexcept -> Int
//////{
//////  for (Int i = size(); i > 0; --i) {
//////    if (data()[i - 1] == c) {
//////      return i - 1;
//////    }
//////  }
//////  return npos;
//////}
//
} // namespace um2
