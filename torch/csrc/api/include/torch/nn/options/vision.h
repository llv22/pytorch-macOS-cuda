#pragma once

#include <torch/arg.h>
#include <torch/csrc/Export.h>
#include <torch/enum.h>
#include <torch/types.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <c10/util/variant.h>
#else
#include <variant>
#endif

namespace torch {
namespace nn {
namespace functional {

/// Options for `torch::nn::functional::grid_sample`.
///
/// Example:
/// ```
/// namespace F = torch::nn::functional;
/// F::grid_sample(input, grid,
/// F::GridSampleFuncOptions().mode(torch::kBilinear).padding_mode(torch::kZeros).align_corners(true));
/// ```
struct TORCH_API GridSampleFuncOptions {
#if defined(__APPLE__) && defined(__MACH__)
  typedef c10::variant<enumtype::kBilinear, enumtype::kNearest> mode_t;
  typedef c10::
      variant<enumtype::kZeros, enumtype::kBorder, enumtype::kReflection>
          padding_mode_t;
#else
  typedef std::variant<enumtype::kBilinear, enumtype::kNearest> mode_t;
  typedef std::
      variant<enumtype::kZeros, enumtype::kBorder, enumtype::kReflection>
          padding_mode_t;
#endif

  /// interpolation mode to calculate output values. Default: Bilinear
  TORCH_ARG(mode_t, mode) = torch::kBilinear;
  /// padding mode for outside grid values. Default: Zeros
  TORCH_ARG(padding_mode_t, padding_mode) = torch::kZeros;
  /// Specifies perspective to pixel as point. Default: false
  TORCH_ARG(c10::optional<bool>, align_corners) = c10::nullopt;
};

} // namespace functional
} // namespace nn
} // namespace torch
