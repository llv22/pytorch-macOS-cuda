#pragma once
#ifndef C10_UTIL_CPP17_H_
#define C10_UTIL_CPP17_H_

#include <c10/macros/Macros.h>
#include <cstdlib>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

#if !defined(__clang__) && !defined(_MSC_VER) && defined(__GNUC__) && \
    __GNUC__ < 5
#error \
    "You're trying to build PyTorch with a too old version of GCC. We need GCC 5 or later."
#endif

#if defined(__clang__) && __clang_major__ < 4
#error \
    "You're trying to build PyTorch with a too old version of Clang. We need Clang 4 or later."
#endif

#if ((defined(_MSC_VER) && (!defined(_MSVC_LANG) || _MSVC_LANG < 201703L)) || \
    (!defined(_MSC_VER) && __cplusplus < 201703L)) && !defined(__APPLE__)
#error You need C++17 to compile PyTorch
#endif

#if defined(_WIN32) && (defined(min) || defined(max))
#error Macro clash with min and max -- define NOMINMAX when compiling your program on Windows
#endif

/*
 * This header adds some polyfills with C++17 functionality
 */

namespace c10 {

// in c++17 std::result_of has been superceded by std::invoke_result.  Since
// c++20, std::result_of is removed.
template <typename F, typename... args>
#if defined(__cpp_lib_is_invocable) && __cpp_lib_is_invocable >= 201703L
using invoke_result = typename std::invoke_result<F, args...>;
#else
using invoke_result = typename std::result_of<F && (args && ...)>;
#endif

template <typename F, typename... args>
using invoke_result_t = typename invoke_result<F, args...>::type;

// std::is_pod is deprecated in C++20, std::is_standard_layout and
// std::is_trivial are introduced in C++11, std::conjunction has been introduced
// in C++17.
template <typename T>
#if defined(__cpp_lib_logical_traits) && __cpp_lib_logical_traits >= 201510L
using is_pod = std::conjunction<std::is_standard_layout<T>, std::is_trivial<T>>;
#else
using is_pod = std::is_pod<T>;
#endif

template <typename T>
constexpr bool is_pod_v = is_pod<T>::value;

namespace guts {

template <typename Base, typename Child, typename... Args>
typename std::enable_if<
    !std::is_array<Base>::value && !std::is_array<Child>::value &&
        std::is_base_of<Base, Child>::value,
    std::unique_ptr<Base>>::type
make_unique_base(Args&&... args) {
  return std::unique_ptr<Base>(new Child(std::forward<Args>(args)...));
}

#if defined(__cpp_lib_logical_traits) && !(defined(_MSC_VER) && _MSC_VER < 1920)

template <class... B>
using conjunction = std::conjunction<B...>;
template <class... B>
using disjunction = std::disjunction<B...>;
template <bool B>
using bool_constant = std::bool_constant<B>;
template <class B>
using negation = std::negation<B>;

#else

// Implementation taken from http://en.cppreference.com/w/cpp/types/conjunction
template <class...>
struct conjunction : std::true_type {};
template <class B1>
struct conjunction<B1> : B1 {};
template <class B1, class... Bn>
struct conjunction<B1, Bn...>
    : std::conditional_t<bool(B1::value), conjunction<Bn...>, B1> {};

// Implementation taken from http://en.cppreference.com/w/cpp/types/disjunction
template <class...>
struct disjunction : std::false_type {};
template <class B1>
struct disjunction<B1> : B1 {};
template <class B1, class... Bn>
struct disjunction<B1, Bn...>
    : std::conditional_t<bool(B1::value), B1, disjunction<Bn...>> {};

// Implementation taken from
// http://en.cppreference.com/w/cpp/types/integral_constant
template <bool B>
using bool_constant = std::integral_constant<bool, B>;

// Implementation taken from http://en.cppreference.com/w/cpp/types/negation
template <class B>
struct negation : bool_constant<!bool(B::value)> {};

#endif

#ifdef __cpp_lib_void_t

template <class T>
using void_t = std::void_t<T>;

#else

// Implementation taken from http://en.cppreference.com/w/cpp/types/void_t
// (it takes CWG1558 into account and also works for older compilers)
template <typename... Ts>
struct make_void {
  typedef void type;
};
template <typename... Ts>
using void_t = typename make_void<Ts...>::type;

#endif

#if defined(USE_ROCM)
// rocm doesn't like the C10_HOST_DEVICE
#define CUDA_HOST_DEVICE
#else
#define CUDA_HOST_DEVICE C10_HOST_DEVICE
#endif

#if defined(__cpp_lib_apply) && !defined(__CUDA_ARCH__)

template <class F, class Tuple>
CUDA_HOST_DEVICE inline constexpr decltype(auto) apply(F&& f, Tuple&& t) {
  return std::apply(std::forward<F>(f), std::forward<Tuple>(t));
}

#else

// Implementation from http://en.cppreference.com/w/cpp/utility/apply (but
// modified)
// TODO This is an incomplete implementation of std::apply, not working for
// member functions.
namespace detail {
template <class F, class Tuple, std::size_t... INDEX>
#if defined(_MSC_VER)
// MSVC has a problem with the decltype() return type, but it also doesn't need
// it
C10_HOST_DEVICE constexpr auto apply_impl(
    F&& f,
    Tuple&& t,
    std::index_sequence<INDEX...>)
#else
// GCC/Clang need the decltype() return type
CUDA_HOST_DEVICE constexpr decltype(auto) apply_impl(
    F&& f,
    Tuple&& t,
    std::index_sequence<INDEX...>)
#endif
{
  return std::forward<F>(f)(std::get<INDEX>(std::forward<Tuple>(t))...);
}
} // namespace detail

template <class F, class Tuple>
CUDA_HOST_DEVICE constexpr decltype(auto) apply(F&& f, Tuple&& t) {
  return detail::apply_impl(
      std::forward<F>(f),
      std::forward<Tuple>(t),
      std::make_index_sequence<
          std::tuple_size<std::remove_reference_t<Tuple>>::value>{});
}

#endif

#undef CUDA_HOST_DEVICE

template <typename Functor, typename... Args>
typename std::enable_if<
    std::is_member_pointer<typename std::decay<Functor>::type>::value,
    typename c10::invoke_result_t<Functor, Args...>>::type
invoke(Functor&& f, Args&&... args) {
  return std::mem_fn(std::forward<Functor>(f))(std::forward<Args>(args)...);
}

template <typename Functor, typename... Args>
typename std::enable_if<
    !std::is_member_pointer<typename std::decay<Functor>::type>::value,
    typename c10::invoke_result_t<Functor, Args...>>::type
invoke(Functor&& f, Args&&... args) {
  return std::forward<Functor>(f)(std::forward<Args>(args)...);
}

namespace detail {
struct _identity final {
  template <class T>
  using type_identity = T;

  template <class T>
  decltype(auto) operator()(T&& arg) {
    return std::forward<T>(arg);
  }
};

template <class Func, class Enable = void>
struct function_takes_identity_argument : std::false_type {};
#if defined(_MSC_VER)
// For some weird reason, MSVC shows a compiler error when using guts::void_t
// instead of std::void_t. But we're only building on MSVC versions that have
// std::void_t, so let's just use that one.
template <class Func>
struct function_takes_identity_argument<
    Func,
    std::void_t<decltype(std::declval<Func>()(_identity()))>> : std::true_type {
};
#else
template <class Func>
struct function_takes_identity_argument<
    Func,
    void_t<decltype(std::declval<Func>()(_identity()))>> : std::true_type {};
#endif
} // namespace detail

} // namespace guts
} // namespace c10

#endif // C10_UTIL_CPP17_H_
