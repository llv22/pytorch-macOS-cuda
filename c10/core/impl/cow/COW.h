#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/intrusive_ptr.h>

namespace c10 {
struct StorageImpl;
class DataPtr;
}; // namespace c10

#if defined(__APPLE__) && defined(__MACH__)
#include <c10/util/Optional.h>
namespace std {
  // Define is_nothrow_move_assignable_v for C++ versions before C++17 where it might not be available.
  using ::c10::optional;
}
#endif

namespace c10::impl::cow {

// Creates a Copy-on-write (COW) clone of the given storage. This will also
// convert the given storage into a COW storage if it is not COW already.
//
// Converting the storage into a COW storage will not be successful if the
// storage's DataPtr has some context (`DataPtr::get_context()`) which is not
// equal to the data pointer (`DataPtr::get()`). In this case, a nullptr is
// returned.
C10_API c10::intrusive_ptr<StorageImpl> lazy_clone_storage(
    StorageImpl& storage);

// Check if a storage has a simple DataPtr with no abnormal context
C10_API bool has_simple_data_ptr(const c10::StorageImpl& storage);

// Check if a DataPtr is COW
C10_API bool is_cow_data_ptr(const c10::DataPtr& data_ptr);

} // namespace c10::impl::cow
