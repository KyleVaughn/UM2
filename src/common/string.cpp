#include <um2/common/string.hpp>

namespace um2
{

// --------------------------------------------------------------------------
// Constructors
// --------------------------------------------------------------------------

UM2_HOSTDEV String::String(String const & s)
    : _size{s._size},
      _capacity{bit_ceil(s._size + 1)},
      _data{new char[static_cast<ulen_t>(bit_ceil(s._size + 1))]}
{
  memcpy(this->_data, s._data, static_cast<size_t>(s._size) + 1);
}

// Turn off warning about member initializers since we need to delete _data before
// assigning a new value to it.
// NOLINTBEGIN(cppcoreguidelines-prefer-member-initializer)
UM2_HOSTDEV String::String(String && s) noexcept
    : _size{s._size},
      _capacity{s._capacity},
      _data{s._data}
{
  s._size = 0;
  s._capacity = 0;
  s._data = nullptr;
}
// NOLINTEND(cppcoreguidelines-prefer-member-initializer)

String::String(std::string const & s)
    : _size{static_cast<len_t>(s.size())},
      _capacity{static_cast<len_t>(bit_ceil(s.size() + 1))},
      _data{new char[bit_ceil(s.size() + 1)]}
{
  memcpy(this->_data, s.data(), s.size() + 1);
}

// --------------------------------------------------------------------------
// Operators
// --------------------------------------------------------------------------

UM2_HOSTDEV auto String::operator=(String const & s) -> String &
{
  if (this != &s) {
    len_t const sizep1 = s._size + 1;
    if (this->_capacity < sizep1) {
      delete[] this->_data;
      this->_capacity = bit_ceil(sizep1);
      this->_data = new char[static_cast<ulen_t>(bit_ceil(sizep1))];
    }
    this->_size = s._size;
    memcpy(this->_data, s._data, static_cast<size_t>(sizep1));
  }
  return *this;
}

UM2_HOSTDEV auto String::operator=(String && s) noexcept -> String &
{
  if (this != &s) {
    delete[] this->_data;
    this->_data = s._data;
    this->_size = s._size;
    this->_capacity = s._capacity;
    s._size = 0;
    s._capacity = 0;
    s._data = nullptr;
  }
  return *this;
}

auto String::operator=(std::string const & s) -> String &
{
  size_t const sizep1 = s.size() + 1;
  if (this->_capacity < static_cast<len_t>(sizep1)) {
    delete[] this->_data;
    this->_capacity = static_cast<len_t>(bit_ceil(sizep1));
    this->_data = new char[bit_ceil(sizep1)];
  }
  this->_size = static_cast<len_t>(s.size());
  memcpy(this->_data, s.data(), s.size() + 1);
  return *this;
}

UM2_PURE auto String::operator==(std::string const & s) const noexcept -> bool
{
  return this->compare(s.data()) == 0;
}

// --------------------------------------------------------------------------
// Methods
// --------------------------------------------------------------------------

// Turn off warning about reinterpret_cast since we need to compare char
// as unsigned char/char
#ifdef __CUDA_ARCH__
UM2_PURE UM2_DEVICE auto String::compare(char const * const s) const noexcept -> int
{
  char const * s1 = this->_data;
  char const * s2 = s;
  while (*s1 && (*s1 == *s2)) {
    ++s1;
    ++s2;
  }
  return *reinterpret_cast<unsigned char const *>(s1) -
         *reinterpret_cast<unsigned char const *>(s2);
}
#else
UM2_PURE UM2_HOST auto String::compare(char const * const s) const noexcept -> int
{
  return strcmp(this->_data, s);
}
#endif

UM2_PURE UM2_HOSTDEV auto String::compare(String const & s) const noexcept -> int
{
  return this->compare(s._data);
}

UM2_PURE auto toString(String const & s) -> std::string
{
  return {s.data(), static_cast<size_t>(s.size())};
}

} // namespace um2
