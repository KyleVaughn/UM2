namespace um2
{

//// -- Accessors --
//
// UM2_NDEBUG_PURE UM2_HOSTDEV constexpr
// char & String::operator [] (length_t const i) {
//    UM2_ASSERT(0 <= i && i < this->_size);
//    return this->_data[i];
//}
//
// UM2_NDEBUG_PURE UM2_HOSTDEV constexpr
// char const & String::operator [] (length_t const i) const {
//    UM2_ASSERT(0 <= i && i < this->_size);
//    return this->_data[i];
//}
//
// UM2_PURE UM2_HOSTDEV constexpr
// char * String::begin() const { return this->_data; }
//
// UM2_PURE UM2_HOSTDEV constexpr
// char * String::end() const { return this->_data + this->_size; }
//
// UM2_PURE UM2_HOSTDEV constexpr
// char const * String::cbegin() const { return this->_data; }
//
// UM2_PURE UM2_HOSTDEV constexpr
// char const * String::cend() const { return this->_data + this->_size; }
//
// UM2_PURE UM2_HOSTDEV constexpr
// length_t String::size() const { return this->_size; }
//
// UM2_PURE UM2_HOSTDEV constexpr
// length_t String::capacity() const { return this->_capacity; }
//
// UM2_PURE UM2_HOSTDEV constexpr
// char * String::data() { return this->_data; }
//
// UM2_PURE UM2_HOSTDEV constexpr
// char const * String::data() const { return this->_data; }
//
// UM2_NDEBUG_PURE UM2_HOSTDEV constexpr
// char & String::front()
//{
//    UM2_ASSERT(this->_size > 0);
//    return this->_data[0];
//}
//
// UM2_NDEBUG_PURE UM2_HOSTDEV constexpr
// char const & String::front() const
//{
//    UM2_ASSERT(this->_size > 0);
//    return this->_data[0];
//}
//
// UM2_NDEBUG_PURE UM2_HOSTDEV constexpr
// char & String::back()
//{
//    UM2_ASSERT(this->_size > 0);
//    return this->_data[this->_size - 1];
//}
//
// UM2_NDEBUG_PURE UM2_HOSTDEV constexpr
// char const & String::back() const
//{
//    UM2_ASSERT(this->_size > 0);
//    return this->_data[this->_size - 1];
//}
//
//// -- Member functions --
//
// UM2_HOSTDEV inline
// void String::reserve(length_t n) {
//    // Reserve one more than n to account for the null terminator
//    if (this->_capacity < n + 1) {
//        // since n > 0, we are safe to cast to unsigned and use bit_ceil
//        // to find the next power of 2
//        n = static_cast<length_t>(bit_ceil(n + 1));
//        char * new_data = new char[static_cast<size_t>(n)];
//        memcpy(new_data, this->_data, static_cast<size_t>(this->_size));
//        memset(new_data + this->_size, '\0', static_cast<size_t>(n - this->_size));
//        delete[] this->_data;
//        this->_data = new_data;
//        this->_capacity = n;
//    }
//}
//
// UM2_PURE UM2_HOSTDEV constexpr
// bool String::empty() const { return this->_size == 0; }
//
//
// UM2_PURE UM2_HOSTDEV constexpr
// bool String::contains(char const value) const
//{
//    for (length_t i = 0; i < this->_size; ++i) {
//        if (this->_data[i] == value) {
//            return true;
//        }
//    }
//    return false;
//}
//
//// -- Constructors --
//
// template <size_t N>
// UM2_HOSTDEV String::String(char const (&s)[N])
//{
//    this->_size = N - 1;
//    this->_capacity = N;
//    this->_data = new char[N];
//    memcpy(this->_data, s, N);
//}
//
//// -- Operators --
//
// UM2_PURE UM2_HOSTDEV constexpr
// bool String::operator == (String const & v) const {
//    if (this->_size != v._size) { return false; }
//    for (length_t i = 0; i < this->_size; ++i) {
//        if (this->_data[i] != v._data[i]) { return false; }
//    }
//    return true;
//}
//
// template <size_t N>
// UM2_HOSTDEV String & String::operator = (char const (&s)[N])
//{
//    if (this->_capacity < static_cast<length_t>(N)) {
//        delete[] this->_data;
//        this->_data = new char[bit_ceil(N)];
//        memset(this->_data, '\0', bit_ceil(N));
//        this->_capacity = static_cast<length_t>(bit_ceil(N));
//    }
//    this->_size = static_cast<length_t>(N - 1);
//    memcpy(this->_data, s, N - 1);
//    return *this;
//}
//
// UM2_PURE UM2_HOSTDEV constexpr bool String::operator < (String const & v) const
//{
//    // Lexicographical comparison.
//    std::string_view const s1(this->_data, static_cast<size_t>(this->_size));
//    std::string_view const s2(v._data, static_cast<size_t>(v._size));
//    return s1 < s2;
//}
//
// template <size_t N>
// UM2_PURE UM2_HOSTDEV constexpr bool String::operator == (char const (&s)[N]) const
//{
//    if (this->_size != static_cast<length_t>(N - 1)) { return false; }
//    for (length_t i = 0; i < this->_size; ++i) {
//        if (this->_data[i] != s[i]) { return false; }
//    }
//    return true;
//}
//
// UM2_PURE constexpr bool String::operator == (std::string const & s) const
//{
//    if (this->_size != static_cast<length_t>(s.size())) { return false; }
//    for (length_t i = 0; i < this->_size; ++i) {
//        if (this->_data[i] != s[static_cast<size_t>(i)]) { return false; }
//    }
//    return true;
//}
//
//// -- Methods --
//
// UM2_PURE constexpr bool String::starts_with(std::string const & s) const
//{
//    if (this->_size < static_cast<length_t>(s.size())) { return false; }
//    for (length_t i = 0; i < static_cast<length_t>(s.size()); ++i) {
//        if (this->_data[i] != s[static_cast<size_t>(i)]) { return false; }
//    }
//    return true;
//}
//
// UM2_PURE constexpr bool String::ends_with(std::string const & s) const
//{
//    length_t const ssize = static_cast<length_t>(s.size());
//    length_t const vsize = this->_size;
//    if (vsize < ssize) { return false; }
//    for (length_t i = 0; i < ssize; ++i) {
//        if (this->_data[vsize - 1 - i] != s[s.size() - 1 - static_cast<size_t>(i)]) {
//            return false;
//        }
//    }
//    return true;
//}

} // namespace um2
