#include <c10/core/impl/cow/COWDeleter.h>
#include <c10/util/Exception.h>
#include <mutex>

namespace c10::impl {

void cow::cow_deleter(void* ctx) {
  static_cast<cow::COWDeleterContext*>(ctx)->decrement_refcount();
}

cow::COWDeleterContext::COWDeleterContext(
    std::unique_ptr<void, DeleterFnPtr> data)
    : data_(std::move(data)) {
  // We never wrap a COWDeleterContext.
  TORCH_INTERNAL_ASSERT(data_.get_deleter() != cow::cow_deleter);
}

auto cow::COWDeleterContext::increment_refcount() -> void {
  auto refcount = ++refcount_;
  TORCH_INTERNAL_ASSERT(refcount > 1);
}

auto cow::COWDeleterContext::decrement_refcount()
    -> std::variant<NotLastReference, LastReference> {
  auto refcount = --refcount_;
  TORCH_INTERNAL_ASSERT(refcount >= 0, refcount);
  if (refcount == 0) {
#if defined(__APPLE__) && defined(__MACH__)
    std::unique_lock<std::shared_mutex> lock(mutex_);
#else
    std::unique_lock lock(mutex_);
#endif
    auto result = std::move(data_);
    lock.unlock();
    delete this;
    return {std::move(result)};
  }

#if defined(__APPLE__) && defined(__MACH__)
  return std::unique_lock<std::shared_mutex>(mutex_);
#else
  return std::shared_lock(mutex_);
#endif
}

cow::COWDeleterContext::~COWDeleterContext() {
  TORCH_INTERNAL_ASSERT(refcount_ == 0);
}

} // namespace c10::impl
