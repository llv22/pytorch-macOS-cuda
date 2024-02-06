# Migration note

```bash
export CXXFLAGS=-D_LIBCPP_DISABLE_AVAILABILITY
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
MAGMA_HOME="/usr/local/lib/magma2.6.1-cu101" MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ USE_LIBUV=1 USE_DISTRIBUTED=ON USE_MPI=ON USE_TENSORPIPE=ON USE_GLOO=ON USE_CUDA_MPI=ON python setup.py clean # prepare
MAGMA_HOME="/usr/local/lib/magma2.6.1-cu101" MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ USE_LIBUV=1 USE_DISTRIBUTED=ON USE_MPI=ON USE_TENSORPIPE=ON USE_GLOO=ON USE_CUDA_MPI=ON python setup.py bdist_wheel
```


## 1, Missing ATen cuda

/usr/local/bin/ccache /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++ -DHAVE_MMAP=1 -DHAVE_SHM_OPEN=1 -DHAVE_SHM_UNLINK=1 -DIDEEP_USE_MKL -DMINIZ_DISABLE_ZIP_READER_CRC32_CHECKS -DONNXIFI_ENABLE_EXT=1 -DONNX_ML=1 -DONNX_NAMESPACE=onnx_torch -DUSE_CUDA_MPI=1 -DUSE_EXTERNAL_MZCRC -D_FILE_OFFSET_BITS=64 -Dcaffe2_nvrtc_EXPORTS -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/build/aten/src -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/aten/src -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/build -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/cmake/../third_party/benchmark/include -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/third_party/onnx -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/build/third_party/onnx -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/third_party/foxi -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/build/third_party/foxi -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/build/third_party/gloo -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/cmake/../third_party/gloo -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/cmake/../third_party/tensorpipe/third_party/libuv/include -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/cmake/../third_party/googletest/googlemock/include -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/cmake/../third_party/googletest/googletest/include -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/third_party/protobuf/src -isystem /Users/llv23/opt/miniconda3/include -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/third_party/gemmlowp -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/third_party/neon2sse -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/third_party/XNNPACK/include -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/third_party/ittapi/include -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/cmake/../third_party/eigen -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/cmake/../third_party/cub -isystem /usr/local/include -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/third_party/ideep/include -D_LIBCPP_DISABLE_AVAILABILITY -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOCUPTI -DLIBKINETO_NOROCTRACER -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=braced-scalar-init -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wvla-extension -Wnewline-eof -Winconsistent-missing-override -Winconsistent-missing-destructor-override -Wno-pass-failed -Wno-error=pedantic -Wno-error=old-style-cast -Wno-error=inconsistent-missing-override -Wno-error=inconsistent-missing-destructor-override -Wconstant-conversion -Wno-invalid-partial-specialization -Wno-aligned-allocation-unavailable -Wno-missing-braces -Qunused-arguments -fcolor-diagnostics -faligned-new -fno-math-errno -fno-trapping-math -Werror=format -Wno-unused-private-field -Wno-missing-braces -DHAVE_AVX2_CPU_DEFINITION -O3 -DNDEBUG -DNDEBUG -std=gnu++14 -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.14.sdk -mmacosx-version-min=10.9 -fPIC -DMKL_HAS_SBGEMM -DTORCH_USE_LIBUV -DCAFFE2_USE_GLOO -MD -MT caffe2/CMakeFiles/caffe2_nvrtc.dir/__/aten/src/ATen/cuda/nvrtc_stub/ATenNVRTC.cpp.o -MF caffe2/CMakeFiles/caffe2_nvrtc.dir/__/aten/src/ATen/cuda/nvrtc_stub/ATenNVRTC.cpp.o.d -o caffe2/CMakeFiles/caffe2_nvrtc.dir/__/aten/src/ATen/cuda/nvrtc_stub/ATenNVRTC.cpp.o -c /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/aten/src/ATen/cuda/nvrtc_stub/ATenNVRTC.cpp

## 2, Migrating from c10 to std

#if defined(__APPLE__) && defined(__MACH__)
#include <c10/util/variant.h>
namespace std {
  using ::c10::variant;
  using ::c10::holds_alternative;
  using ::c10::get;
  // https://stackoverflow.com/questions/56843413/stdbyte-is-not-member-of-std
  enum class byte : unsigned char {};
}// namespace std
#else
#include <variant>
#endif


#if defined(__APPLE__) && defined(__MACH__)
#include <c10/util/Optional.h>
namespace std {
  using c10::optional;
}//namespace
#else
#include <optional>
#endif

#if defined(__APPLE__) && defined(__MACH__)
#include <c10/util/variant.h>
#endif

#if defined(__APPLE__) && defined(__MACH__)
#include <c10/util/variant.h>
#else
#include <variant>
#endif

#if defined(__APPLE__) && defined(__MACH__)
c10::visit
#else
#endif 

MetadataShape compute_variant_shape(const at::Tensor& input) {
  if (input.is_nested() && !input.unsafeGetTensorImpl()->is_python_dispatch()) {
    auto nested_size = input._nested_tensor_size();
#if defined(__APPLE__) && defined(__MACH__)
    return MetadataShape{c10::in_place_type<at::Tensor>, nested_size};
#else
    return MetadataShape{std::in_place_type<at::Tensor>, nested_size};
#endif
  }
#if defined(__APPLE__) && defined(__MACH__)
  return MetadataShape{c10::in_place_type<SymIntSmallVec>, input.sym_sizes()};
#else
  return MetadataShape{std::in_place_type<SymIntSmallVec>, input.sym_sizes()};
#endif
}
