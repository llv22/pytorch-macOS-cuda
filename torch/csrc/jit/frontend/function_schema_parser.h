#pragma once

#include <ATen/core/function_schema.h>
#include <c10/macros/Macros.h>
#include <string>
#if defined(__APPLE__) && defined(__MACH__)
#include <c10/util/variant.h>
namespace std {
  using ::c10::variant;
  using ::c10::get;
} // namespace std
#else
#include <variant>
#endif

namespace torch {
namespace jit {

TORCH_API std::variant<c10::OperatorName, c10::FunctionSchema> parseSchemaOrName(
    const std::string& schemaOrName);
TORCH_API c10::FunctionSchema parseSchema(const std::string& schema);
TORCH_API c10::OperatorName parseName(const std::string& name);

} // namespace jit
} // namespace torch
