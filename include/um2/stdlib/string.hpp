#pragma once

#include <um2/config.hpp>

#include <um2/stdlib/assert.hpp>
#include <um2/stdlib/algorithm/copy.hpp>
//#include <um2/stdlib/math.hpp>      // min
#include <um2/stdlib/memory/addressof.hpp> // addressof
#include <um2/stdlib/utility/move.hpp>     // move
//
//#include <cstring> // memcpy, strcmp
//#include <string>  // std::string

// The length of a string, not including the null terminator.
inline constexpr auto
length(char const * s) -> Int
{
  Int n = 0;
  while (s[n] != '\0') {
    ++n;
  }
  return n;
}

//==============================================================================
// STRING
//==============================================================================
// An std::string-like class, but without an allocator template parameter.
// Uses small string optimization.
//
// NOTE: ASSUMES LITTLE ENDIAN
// This should be true for all x86 and Apple processors and both NVIDIA and AMD GPUs.
//
// For developers: clang-tidy is a bit overzealous with its warnings in this file.

// NOLINTBEGIN(clang-analyzer-cplusplus.NewDelete)

namespace um2
{

// NOLINTNEXTLINE(cppcoreguidelines-special-member-functions)
class String
{

  using Ptr = char *;
  using ConstPtr = char const *;

  // Heap-allocated string representation.
  // 24 bytes
  struct Long {
    Ptr data;             // Pointer to the string data.
    uint64_t size;        // Size of the string. (Does NOT include null.)
    uint64_t cap : 63;    // Capacity of the string. (Does include null.)
    uint64_t is_long : 1; // Single bit for representation flag.
  };

  // The maximum capacity of a short string.
  // 24 byte string - 1 byte for representation flag and size = 23 bytes
  // This includes the null terminator.
  static uint64_t constexpr min_cap = sizeof(Long) - 1;

  // Stack-allocated string representation.
  struct Short {
    char data[min_cap];  // Data of the string.
    uint8_t size : 7;    // 7 bits for the size of the string. (Does not include null.)
    uint8_t is_long : 1; // Single bit for representation flag.
  };

  // Mask to convert to a 63-bit unsigned integer.
  static uint64_t constexpr long_cap_mask = 0x7FFFFFFFFFFFFFFF;

  // Mask to convert to a 7-bit unsigned integer.
  static uint8_t constexpr short_size_mask = 0x7F;

  // Raw representation of the string.
  // For the purpose of copying and moving.
  struct Raw {
    uint64_t raw[3];
  };

  // Union of all representations.
  struct Rep {
    union {
      Short s;
      Long l;
      Raw r;
    };
  };

  Rep _r;

  //==============================================================================
  // Private functions
  //==============================================================================
  // NOLINTBEGIN(readability-identifier-naming) match std::string

  HOSTDEV inline constexpr void
  clear() noexcept;

  HOSTDEV constexpr void
  clearAndShrink() noexcept;

//  HOSTDEV constexpr void
//  copyAssignAlloc(String const & s) noexcept;

  // Does a string of length n fit in a short string? n does not include the null terminator.
  CONST HOSTDEV static constexpr auto
  fitsInShort(uint64_t n) noexcept -> bool;

  // Get the capacity of the long string. Includes the null terminator.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getLongCap() const noexcept -> uint64_t;

  // Get a pointer to the long string data.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getLongPointer() noexcept -> Ptr;

  // Get a pointer to the long string data.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getLongPointer() const noexcept -> ConstPtr;

  // Get the size of the long string. Does NOT include the null terminator.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getLongSize() const noexcept -> uint64_t;

  // Get a pointer to the string data regardless of representation.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getPointer() noexcept -> Ptr; 

  // Get a pointer to the string data regardless of representation.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getPointer() const noexcept -> ConstPtr;

  // Get a pointer to the short string data.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getShortPointer() noexcept -> Ptr;

  // Get a pointer to the short string data.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getShortPointer() const noexcept -> ConstPtr;

  // Get the size of the short string. Does NOT include the null terminator.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  getShortSize() const noexcept -> uint64_t;

  // Initialize the string with a pointer to a string and its length. 
  // Does not include the null terminator.
  HOSTDEV constexpr void
  init(ConstPtr s, uint64_t size) noexcept;

  HOSTDEV constexpr void
  setLongCap(uint64_t cap) noexcept;

  HOSTDEV constexpr void
  setLongPointer(Ptr p) noexcept;

  HOSTDEV constexpr void
  setLongSize(uint64_t size) noexcept;

  HOSTDEV constexpr void
  setShortSize(uint64_t size) noexcept;

public:
  // The maximum capacity of a long string.
  static Int constexpr npos = intMax();

  //==============================================================================
  // Constructors
  //==============================================================================

  HOSTDEV constexpr String() noexcept;

  HOSTDEV constexpr String(String const & s) noexcept;

  HOSTDEV constexpr String(String && s) noexcept;

//  template <uint64_t N>
//  HOSTDEV constexpr String(char const (&s)[N]) noexcept;

  // NOLINTBEGIN(google-explicit-constructor) match std::string
  HOSTDEV constexpr String(char const * s) noexcept;
  // NOLINTEND(google-explicit-constructor)
//
//  HOSTDEV constexpr String(char const * begin, char const * end) noexcept;
//
//  HOSTDEV constexpr String(char const * s, Int n) noexcept;
//
//  template <std::integral T>
//  explicit constexpr String(T x) noexcept;
//
//  template <std::floating_point T>
//  explicit constexpr String(T x) noexcept;
//
  //==============================================================================
  // Destructor
  //==============================================================================

  HOSTDEV inline constexpr ~String() noexcept;

  //==============================================================================
  // Accessors
  //==============================================================================

  // The number of characters that can be held without reallocating storage.
  // Does not include the null terminator.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  capacity() const noexcept -> Int;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  data() noexcept -> Ptr;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  data() const noexcept -> ConstPtr;

  PURE HOSTDEV [[nodiscard]] constexpr auto
  isLong() const noexcept -> bool;

  // Not including the null terminator.
  PURE HOSTDEV [[nodiscard]] constexpr auto
  size() const noexcept -> Int;

  //==============================================================================
  // Operators
  //==============================================================================

//  HOSTDEV constexpr auto
//  operator=(String const & s) noexcept -> String &;

  HOSTDEV constexpr auto
  operator=(String && s) noexcept -> String &;

//  constexpr auto
//  operator=(std::string const & s) noexcept -> String &;
//
//  constexpr auto
//  operator=(std::string && s) noexcept -> String &;
//
//  template <uint64_t N>
//  HOSTDEV constexpr auto
//  operator=(char const (&s)[N]) noexcept -> String &;
//
//  PURE HOSTDEV constexpr auto
//  operator==(String const & s) const noexcept -> bool;
//
//  PURE HOSTDEV constexpr auto
//  operator!=(String const & s) const noexcept -> bool;
//
//  PURE HOSTDEV constexpr auto
//  operator<(String const & s) const noexcept -> bool;
//
//  PURE HOSTDEV constexpr auto
//  operator<=(String const & s) const noexcept -> bool;
//
//  PURE HOSTDEV constexpr auto
//  operator>(String const & s) const noexcept -> bool;
//
//  PURE HOSTDEV constexpr auto
//  operator>=(String const & s) const noexcept -> bool;
//
//  PURE HOSTDEV constexpr auto
//  operator[](Int i) noexcept -> char &;
//
//  PURE HOSTDEV constexpr auto
//  operator[](Int i) const noexcept -> char const &;
//
//  HOSTDEV constexpr auto
//  operator+=(String const & s) noexcept -> String &;
//
//  HOSTDEV constexpr auto
//  operator+=(char c) noexcept -> String &;
//
//  //==============================================================================
//  // Methods
//  //==============================================================================
//
//  PURE HOSTDEV [[nodiscard]] constexpr auto
//  compare(String const & s) const noexcept -> int;
//
//  PURE HOSTDEV [[nodiscard]] constexpr auto
//  c_str() const noexcept -> char const *;
//
//  PURE HOSTDEV [[nodiscard]] constexpr auto
//  ends_with(String const & s) const noexcept -> bool;
//
//  template <uint64_t N>
//  PURE HOSTDEV [[nodiscard]] constexpr auto
//  ends_with(char const (&s)[N]) const noexcept -> bool;
//
//  PURE HOSTDEV [[nodiscard]] constexpr auto
//  find_last_of(char c) const noexcept -> Int;
//
  PURE HOSTDEV [[nodiscard]] constexpr auto
  starts_with(ConstPtr s) const noexcept -> bool;
//
//  PURE HOSTDEV [[nodiscard]] constexpr auto
//  substr(Int pos, Int len = npos) const -> String;
//
//  // NOLINTEND(readability-identifier-naming)
}; // class String

////==============================================================================
//// Non-member operators
////==============================================================================
//
//HOSTDEV constexpr auto
//operator+(String l, String const & r) noexcept -> String;
//
//template <uint64_t N>
//HOSTDEV constexpr auto
//operator+(String l, char const (&r)[N]) noexcept -> String;
//
//template <uint64_t N>
//HOSTDEV constexpr auto
//operator+(char const (&l)[N], String const & r) noexcept -> String;
//
////==============================================================================
//// Non-member functions
////==============================================================================
//
//template <typename T>
//constexpr auto
//toString(T const & t) noexcept -> String;
//
////==============================================================================
//// MACROS
////==============================================================================
//// To maintain constexpr-ness and readability, we define a few macros to avoid
//// repeating ourselves
////
//// gcc seems to have a bug that causes it to generate a call to memmove that
//// is out of bounds when using std::copy with -O3. Therefore, we write out a basic
//// copy loop instead.
//
/*

//#define COPY_LOOP(first, last, dest)                                                      \
//  auto f = (first);                                                                       \
//  auto l = (last);                                                                        \
//  auto d = (dest);                                                                        \
//  while (f != l) {                                                                        \
//    *d = *f;                                                                              \
//    ++f;                                                                                  \
//    ++d;                                                                                  \
//  }
//
//// mimic allocateAndCopy(char const *, uint64_t)
//#define LONG_ALLOCATE_AND_COPY(ptr, num_elem)                                            \
//  char const * ss = (ptr);                                                               \
//  uint64_t const nn = (num_elem);                                                        \
//  ASSERT_ASSUME(nn > 0);                                                                 \
//  _r.l.is_long = 1;                                                                      \
//  _r.l.cap = (nn - 1) & long_cap_mask;                                                   \
//  _r.l.size = nn - 1;                                                                    \
//  _r.l.data = static_cast<char *>(::operator new(nn));                                   \
//  COPY_LOOP(ss, ss + nn, _r.l.data);
//
//#define SHORT_COPY(ptr, num_elem)                                                        \
//  char const * ss = (ptr);                                                               \
//  uint64_t const nn = (num_elem);                                                        \
//  ASSERT_ASSUME(nn > 0);                                                                 \
//  _r.s.is_long = 0;                                                                      \
//  _r.s.size = (nn - 1) & short_size_mask;                                                \
//  COPY_LOOP(ss, ss + nn, _r.s.data);
//
*/
//==============================================================================
// Private functions
//==============================================================================

HOSTDEV inline constexpr String::~String() noexcept
{
  if (isLong()) {
    ::operator delete(getLongPointer());
  }
}

// Retains the representation of the string, but sets the size to 0.
HOSTDEV inline constexpr void
String::clear() noexcept
{
  if (isLong()) {
    *getLongPointer() = '\0';
    setLongSize(0);
  } else {
    *getShortPointer() = '\0';
    setShortSize(0);
  }
}

HOSTDEV inline constexpr void
String::clearAndShrink() noexcept
{
  clear();
  if (isLong()) {
    ::operator delete(getLongPointer());
    _r = Rep();
  }
}

//HOSTDEV constexpr void
//String::copyAssignAlloc(String const & s) noexcept
//{
//  if (!s.isLong()) {
////    // If s is a short string, we can just copy the whole struct after
////    // clearing the current string.
////    clearAndShrink();
////    _r = s._r;
////  } else {
////    // If s is a long string, we need to allocate new memory.
////    Ptr alloc = static_cast<Ptr>(::operator new(s.getLongCap()));
////    if (isLong()) {
////      ::operator delete(getLongPointer());
////    }
////    setLongPointer(alloc);
////    setLongCap(s.getLongCap());
////    setLongSize(s.getLongSize());
//  }
//}

PURE HOSTDEV constexpr auto
String::getLongSize() const noexcept -> uint64_t
{
  return _r.l.size;
}

PURE HOSTDEV constexpr auto
String::getShortSize() const noexcept -> uint64_t
{
  return _r.s.size;
}

PURE HOSTDEV constexpr auto
String::getLongCap() const noexcept -> uint64_t
{
  return _r.l.cap;
}

PURE HOSTDEV constexpr auto
// NOLINTNEXTLINE(readability-make-member-function-const) we offer const next
String::getLongPointer() noexcept -> Ptr
{
  return _r.l.data;
}

PURE HOSTDEV constexpr auto
String::getLongPointer() const noexcept -> ConstPtr
{
  return _r.l.data;
}

PURE HOSTDEV constexpr auto
String::getShortPointer() noexcept -> Ptr
{
  return um2::addressof(_r.s.data[0]);
}

PURE HOSTDEV constexpr auto
String::getShortPointer() const noexcept -> ConstPtr
{
  return um2::addressof(_r.s.data[0]);
}

PURE HOSTDEV constexpr auto
String::getPointer() noexcept -> Ptr
{
  return isLong() ? getLongPointer() : getShortPointer();
}

PURE HOSTDEV constexpr auto
String::getPointer() const noexcept -> ConstPtr
{
  return isLong() ? getLongPointer() : getShortPointer();
}

// n does not include the null terminator
CONST HOSTDEV constexpr auto
String::fitsInShort(uint64_t n) noexcept -> bool
{
  return n < min_cap;
}

HOSTDEV constexpr void
String::init(ConstPtr s, uint64_t size) noexcept
{
  ASSERT(s != nullptr);
  Ptr p = nullptr;
  if (fitsInShort(size)) {
    setShortSize(size);
    p = getShortPointer();
  } else {
    p = static_cast<Ptr>(::operator new(size + 1));
    setLongPointer(p);
    setLongCap(size + 1);
    setLongSize(size);
  }
  um2::copy(s, s + size, p);
  p[size] = '\0';
}

HOSTDEV constexpr void
String::setLongCap(uint64_t cap) noexcept
{
  _r.l.cap = cap & long_cap_mask;
  _r.l.is_long = true;
}

HOSTDEV constexpr void
String::setLongPointer(Ptr p) noexcept
{
  _r.l.data = p;
}

HOSTDEV constexpr void
String::setLongSize(uint64_t size) noexcept
{
  _r.l.size = size;
}

HOSTDEV constexpr void
String::setShortSize(uint64_t size) noexcept
{
  ASSERT(size < min_cap);
  _r.s.size = size & short_size_mask;
  _r.s.is_long = false;
}

//template <typename T>
//constexpr auto
//toString(T const & t) noexcept -> String
//{
//  return String(t);
//}
//
//HOSTDEV constexpr auto
//operator+(String l, String const & r) noexcept -> String
//{
//  l += r;
//  return l;
//}
//
//template <uint64_t N>
//HOSTDEV constexpr auto
//operator+(String l, char const (&r)[N]) noexcept -> String
//{
//  l += String(r);
//  return l;
//}
//
//template <uint64_t N>
//HOSTDEV constexpr auto
//operator+(char const (&l)[N], String const & r) noexcept -> String
//{
//  String tmp(l);
//  tmp += r;
//  return tmp;
//}
//
//==============================================================================
// Constructors
//==============================================================================

// For a union without a user-defined default constructor, value initialization is zero 
// initialization
HOSTDEV constexpr String::String() noexcept
    : _r()
{
  ASSERT(_r.r.raw[0] == 0);
  ASSERT(_r.r.raw[1] == 0);
  ASSERT(_r.r.raw[2] == 0);
}

HOSTDEV constexpr String::String(String const & s) noexcept
{
  if (!s.isLong()) {
    // If this is a short string, it is trivially copyable
    _r = s._r;
  } else {
    init(s.getLongPointer(), s.getLongSize());
  }
}

HOSTDEV constexpr String::String(String && s) noexcept
    : _r(um2::move(s._r))
{
  // If short string, we can copy trivially
  // If long string, we need to move the data.
  // Since the data is a pointer, we can just copy the pointer.
  // Therefore, either way, we can just copy the whole struct.
  s._r = Rep();
}

////template <uint64_t N>
////HOSTDEV constexpr String::String(char const (&s)[N]) noexcept
////{
////  // N includes the null terminator
////  char const * p = addressof(s[0]);
////  if constexpr (N <= min_cap) {
////    SHORT_COPY(p, N);
////    ASSERT(_r.s.data[N - 1] == '\0');
////  } else {
////    LONG_ALLOCATE_AND_COPY(p, N);
////    ASSERT(_r.l.data[N - 1] == '\0');
////  }
////}

HOSTDEV constexpr String::String(char const * s) noexcept
{
  ASSERT(s != nullptr);
  // Get the length of the string (not including null terminator)
  auto const n = static_cast<uint64_t>(length(s));
  ASSERT(n > 0);
  init(s, n);
}

////HOSTDEV constexpr String::String(char const * s, Int const n) noexcept
////{
////  // Short string
////  auto const cap = static_cast<uint64_t>(n);
////  if (cap + 1 <= min_cap) {
////    SHORT_COPY(s, cap + 1);
////    _r.s.data[cap] = '\0';
////  } else {
////    LONG_ALLOCATE_AND_COPY(s, cap + 1);
////    _r.l.data[cap] = '\0';
////  }
////}
////
////HOSTDEV constexpr String::String(char const * begin, char const * end) noexcept
////{
////  // begin and end are pointers to the first and one past the last character
////  // of the string, respectively.
////  //
////  // "test" -> begin = &t, end = &t + 4
////  // Hence, end is not a valid memory location, nor necessarily the null
////  // terminator.
////  auto const n = static_cast<uint64_t>(end - begin);
////  ASSERT(n > 0);
////  if (n + 1 <= min_cap) {
////    _r.s.is_long = 0;
////    _r.s.size = n & short_size_mask;
////    auto * dest = addressof(_r.s.data[0]); 
////    while (begin != end) {              
////      *dest = *begin;                  
////      ++begin;                        
////      ++dest;                        
////    }
////    *dest = '\0';
////  } else {
////    _r.l.is_long = 1;                
////    _r.l.cap = n & long_cap_mask;   
////    _r.l.size = n;                 
////    _r.l.data = static_cast<char *>(::operator new(n));
////    auto * dest = _r.l.data; 
////    while (begin != end) {              
////      *dest = *begin;                  
////      ++begin;                        
////      ++dest;                        
////    }
////    *dest = '\0';
////  }
////}
////
////// std::to_string should not allocate here due to small string optimization, so
////// we are fine keeping these for now. They should be replaced with something
////// like snprintf in the future.
////template <std::integral T>
////constexpr String::String(T x) noexcept
////{
////  // A 64-bit integer can have at most 20 chars
////  std::string const s = std::to_string(x);
////  auto const cap = s.size();
////  ASSERT_ASSUME(cap < min_cap);
////  SHORT_COPY(s.data(), cap + 1);
////  _r.s.data[cap] = '\0';
////}
////
////template <std::floating_point T>
////constexpr String::String(T x) noexcept
////{
////  // A 64-bit floating point number can have at most 24 chars, but
////  // many numbers will have fewer than 24 chars. So, we ASSERT that
////  // this is the case.
////  std::string const s = std::to_string(x);
////  auto const cap = s.size();
////  ASSERT_ASSUME(cap < min_cap);
////  SHORT_COPY(s.data(), cap + 1);
////  _r.s.data[cap] = '\0';
////}
////
//==============================================================================
// Accessors
//==============================================================================

PURE HOSTDEV constexpr auto
String::isLong() const noexcept -> bool
{
  return _r.s.is_long;
}

PURE HOSTDEV constexpr auto
String::size() const noexcept -> Int
{
  return isLong() ? static_cast<Int>(getLongSize()) : static_cast<Int>(getShortSize());
}

// Allocated bytes - 1 for null terminator
PURE HOSTDEV constexpr auto
String::capacity() const noexcept -> Int
{
  return isLong() ? static_cast<Int>(getLongCap()) - 1 : static_cast<Int>(min_cap) - 1;
}

PURE HOSTDEV constexpr auto
String::data() noexcept -> Ptr
{
  return getPointer();
}

PURE HOSTDEV constexpr auto
String::data() const noexcept -> ConstPtr
{
  return getPointer();
}

//==============================================================================
// Operators
//==============================================================================

//HOSTDEV constexpr auto
//String::operator=(String const & s) noexcept -> String &
//{
//  if (this != um2::addressof(s)) {
//    copyAssignAlloc(s);
//    // if s is a short string, we are done.
//    // Otherwise, we must now copy the data.
//    if (isLong()) {
//      um2::copy(s.getLongPointer(), s.getLongPointer() + s.getLongSize() + 1, getLongPointer());
//    }
//  }
//
//  return *this;
//}

HOSTDEV constexpr auto
String::operator=(String && s) noexcept -> String &
{
  if (this != addressof(s)) {
    if (isLong()) {
      ::operator delete(getLongPointer());
    }
    _r = s._r;
    // We just copied the struct, so we are safe to overwrite s.
    // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks)
    s._r = Rep();
  }
  return *this;
}

////// These std::string assignment operators are a bit inefficient, but the number of
////// heap allocations is the same as if we had just copied the string, so it's not
////// too bad.
////constexpr auto
////String::operator=(std::string const & s) noexcept -> String &
////{
////  String tmp(s.c_str());
////  return *this = um2::move(tmp);
////}
////
////constexpr auto
////String::operator=(std::string && s) noexcept -> String &
////{
////  String tmp(s.c_str());
////  return *this = um2::move(tmp);
////}
////
////template <uint64_t N>
////HOSTDEV constexpr auto
////String::operator=(char const (&s)[N]) noexcept -> String &
////{
////  if (isLong()) {
////    ::operator delete(_r.l.data);
////  }
////  // Short string
////  if constexpr (N <= min_cap) {
////    SHORT_COPY(s, N);
////    ASSERT(_r.s.data[N - 1] == '\0');
////  } else {
////    LONG_ALLOCATE_AND_COPY(s, N);
////    ASSERT(_r.l.data[N - 1] == '\0');
////  }
////  return *this;
////}
////
////PURE HOSTDEV constexpr auto
////String::operator==(String const & s) const noexcept -> bool
////{
////  Int const l_size = size();
////  Int const r_size = s.size();
////  if (l_size != r_size) {
////    return false;
////  }
////  char const * l_data = data();
////  char const * r_data = s.data();
////  for (Int i = 0; i < l_size; ++i) {
////    // NOLINTNEXTLINE
////    if (*l_data != *r_data) {
////      return false;
////    }
////    ++l_data;
////    ++r_data;
////  }
////  return true;
////}
////
////PURE HOSTDEV constexpr auto
////String::operator!=(String const & s) const noexcept -> bool
////{
////  return !(*this == s);
////}
////
////PURE HOSTDEV constexpr auto
////String::operator<(String const & s) const noexcept -> bool
////{
////  return compare(s) < 0;
////}
////
////PURE HOSTDEV constexpr auto
////String::operator<=(String const & s) const noexcept -> bool
////{
////  return compare(s) <= 0;
////}
////
////PURE HOSTDEV constexpr auto
////String::operator>(String const & s) const noexcept -> bool
////{
////  return compare(s) > 0;
////}
////
////PURE HOSTDEV constexpr auto
////String::operator>=(String const & s) const noexcept -> bool
////{
////  return compare(s) >= 0;
////}
////
////PURE HOSTDEV constexpr auto
////String::operator[](Int i) noexcept -> char &
////{
////  return data()[i];
////}
////
////PURE HOSTDEV constexpr auto
////String::operator[](Int i) const noexcept -> char const &
////{
////  return data()[i];
////}
////
////HOSTDEV constexpr auto
////String::operator+=(String const & s) noexcept -> String &
////{
////  auto const new_size = static_cast<uint64_t>(size() + s.size());
////  if (fitsInShort(new_size + 1)) {
////    // This must be a short string, so we can just copy the data.
////    ASSERT(!isLong());
////    ASSERT(!s.isLong());
////    ASSERT(new_size < min_cap);
////    memcpy(getPointer() + size(), s.data(), static_cast<uint64_t>(s.size() + 1));
////    _r.s.size = new_size & short_size_mask;
////  } else {
////    // Otherwise, we need to allocate a new string and copy the data.
////    char * tmp = static_cast<char *>(::operator new(new_size + 1));
////    memcpy(tmp, data(), static_cast<uint64_t>(size()));
////    memcpy(tmp + size(), s.data(), static_cast<uint64_t>(s.size() + 1));
////    if (isLong()) {
////      ::operator delete(_r.l.data);
////    }
////    _r.l.is_long = 1;
////    _r.l.cap = (new_size + 1) & long_cap_mask;
////    _r.l.size = new_size;
////    _r.l.data = tmp;
////  }
////  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks) Valgrind says this is fine
////  return *this;
////}
////
////HOSTDEV constexpr auto
////String::operator+=(char const c) noexcept -> String &
////{
////  // If this is a short string and the size of the new string is less than
////  // the capacity of the short string, we can just append the new string.
////  auto const new_size = static_cast<uint64_t>(size() + 1);
////  if (fitsInShort(new_size + 1)) {
////    ASSERT(!isLong());
////    _r.s.data[size()] = c;
////    _r.s.data[size() + 1] = '\0';
////    _r.s.size += 1;
////  } else {
////    // Otherwise, we need to allocate a new string and copy the data.
////    char * tmp = static_cast<char *>(::operator new(new_size + 1));
////    memcpy(tmp, data(), static_cast<uint64_t>(size()));
////    tmp[size()] = c;
////    tmp[size() + 1] = '\0';
////    if (isLong()) {
////      ::operator delete(_r.l.data);
////    }
////    _r.l.is_long = 1;
////    _r.l.cap = (new_size + 1) & long_cap_mask;
////    _r.l.size = new_size;
////    _r.l.data = tmp;
////  }
////  // NOLINTNEXTLINE(clang-analyzer-cplusplus.NewDeleteLeaks) Valgrind says this is fine
////  return *this;
////}
////
//==============================================================================
// Member functions 
//==============================================================================

////PURE HOSTDEV constexpr auto
////String::compare(String const & s) const noexcept -> int
////{
////  Int const l_size = size();
////  Int const r_size = s.size();
////  Int const min_size = um2::min(l_size, r_size);
////  char const * l_data = data();
////  char const * r_data = s.data();
////  for (Int i = 0; i < min_size; ++i) {
////    if (*l_data != *r_data) {
////      return static_cast<int>(*l_data) - static_cast<int>(*r_data);
////    }
////    ++l_data;
////    ++r_data;
////  }
////  return static_cast<int>(l_size) - static_cast<int>(r_size);
////}
////
////PURE HOSTDEV constexpr auto
////String::c_str() const noexcept -> char const *
////{
////  return data();
////}
////
//PURE HOSTDEV constexpr auto
//String::starts_with(ConstPtr s) const noexcept -> bool
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

////PURE HOSTDEV constexpr auto
////String::ends_with(String const & s) const noexcept -> bool
////{
////  Int const l_size = size();
////  Int const r_size = s.size();
////  if (l_size < r_size) {
////    return false;
////  }
////  char const * l_data = data() + l_size - r_size;
////  char const * r_data = s.data();
////  for (Int i = 0; i < r_size; ++i) {
////    if (*l_data != *r_data) {
////      return false;
////    }
////    ++l_data;
////    ++r_data;
////  }
////  return true;
////}
////
////template <uint64_t N>
////PURE HOSTDEV constexpr auto
////// NOLINTNEXTLINE(readability-identifier-naming) match std::string
////String::ends_with(char const (&s)[N]) const noexcept -> bool
////{
////  return ends_with(String(s));
////}
////
////PURE HOSTDEV constexpr auto
////String::substr(Int pos, Int len) const -> String
////{
////  ASSERT(pos <= size());
////  if (len == npos || pos + len > size()) {
////    len = size() - pos;
////  }
////  return String{data() + pos, len};
////}
////
////PURE HOSTDEV constexpr auto
////String::find_last_of(char const c) const noexcept -> Int
////{
////  for (Int i = size(); i > 0; --i) {
////    if (data()[i - 1] == c) {
////      return i - 1;
////    }
////  }
////  return npos;
////}

} // namespace um2

// NOLINTEND(clang-analyzer-cplusplus.NewDelete)
