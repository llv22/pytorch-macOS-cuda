#pragma once

#include <c10/util/irange.h>
#include <memory>
#include <mutex>

#if defined(__APPLE__) && defined(__MACH__)
#include <type_traits>
// namespace std {
//   // Define is_nothrow_move_assignable_v for C++ versions before C++17 where it might not be available.
//   template <class T>
//   constexpr bool is_standard_layout_v = std::is_standard_layout<T>::value;
// }
#endif

namespace at{ namespace native {

// Hashing machinery for Params
// Fowler–Noll–Vo hash function
// see
// https://en.wikipedia.org/wiki/Fowler%E2%80%93Noll%E2%80%93Vo_hash_function
template <typename Params>
struct ParamsHash {
  // Params must be a POD because we read out its memory
  // contents as char* when hashing
  static_assert(std::is_standard_layout<Params>::value, "Params is not POD");

  size_t operator()(const Params& params) const {
    auto ptr = reinterpret_cast<const uint8_t*>(&params);
    uint32_t value = 0x811C9DC5;
    for (const auto i : c10::irange(sizeof(Params))) {
      value ^= ptr[i];
      value *= 0x01000193;
    }
    return (size_t)value;
  }
};

template <typename Params>
struct ParamsEqual {
  // Params must be a POD because we read out its memory
  // contents as char* when comparing
  static_assert(std::is_standard_layout<Params>::value, "Params is not POD");

  bool operator()(const Params& a, const Params& b) const {
    auto ptr1 = reinterpret_cast<const uint8_t*>(&a);
    auto ptr2 = reinterpret_cast<const uint8_t*>(&b);
    return memcmp(ptr1, ptr2, sizeof(Params)) == 0;
  }
};

// Provide explicit byte-for-byte constructors to avoid uwittingly leaving
// padding bytes unitialized (e.g., when passing Params by value)
template <typename T>
struct ParamsWrapper {
  T pod;
  static_assert(
      std::is_standard_layout<T>::value,
      "ParamsWrapper cannot wrap non-POD data");

  ParamsWrapper() {
    memset(&(this->pod), 0, sizeof(this->pod));
  }

  ParamsWrapper(const ParamsWrapper& other) {
    memcpy(&(this->pod), &(other.pod), sizeof(this->pod));
  }

  ParamsWrapper(ParamsWrapper&& other) noexcept {
    memcpy(&(this->pod), &(other.pod), sizeof(this->pod));
  }

  ParamsWrapper& operator=(const ParamsWrapper& other) {
    memcpy(&(this->pod), &(other.pod), sizeof(this->pod));
    return *this;
  }

  ParamsWrapper& operator=(ParamsWrapper&& other) noexcept {
    memcpy(&(this->pod), &(other.pod), sizeof(this->pod));
    return *this;
  }

  inline friend bool operator==(
      const ParamsWrapper& lhs,
      const ParamsWrapper& rhs) noexcept {
    auto ptr1 = reinterpret_cast<const uint8_t*>(&(lhs.pod));
    auto ptr2 = reinterpret_cast<const uint8_t*>(&(rhs.pod));
    return memcmp(ptr1, ptr2, sizeof(lhs.pod)) == 0;
  }
};

// Wrapped version: this allows the outer struct to have custom copy and move
// constructors for additional safety
template <typename ParamsWrapper>
struct ParamsWrapperHash {
  // Params must be a POD because we read out its memory
  // contents as char* when hashing
  static_assert(
      std::is_standard_layout<decltype(ParamsWrapper::pod)>::value,
      "ParamsWrapper cannot wrap non-POD data");

  size_t operator()(const ParamsWrapper& params_wrapper) const {
    auto ptr = reinterpret_cast<const uint8_t*>(&(params_wrapper.pod));
    uint32_t value = 0x811C9DC5;
    for (const auto i : c10::irange(sizeof(params_wrapper.pod))) {
      value ^= ptr[i];
      value *= 0x01000193;
    }
    return (size_t)value;
  }
};

}} // namespace at::native
