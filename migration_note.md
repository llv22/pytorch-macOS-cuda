# Migration note

Preparation of building library:

```bash
export CXXFLAGS=-D_LIBCPP_DISABLE_AVAILABILITY
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
MAGMA_HOME="/usr/local/lib/magma2.6.1-cu101" MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ USE_LIBUV=1 USE_DISTRIBUTED=ON USE_MPI=ON USE_TENSORPIPE=ON USE_GLOO=ON USE_CUDA_MPI=ON python setup.py clean
MAGMA_HOME="/usr/local/lib/magma2.6.1-cu101" MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ CMAKE_BUILD_TYPE=1 USE_LIBUV=1 USE_DISTRIBUTED=ON USE_MPI=ON USE_TENSORPIPE=ON USE_GLOO=ON USE_CUDA_MPI=ON python setup.py bdist_wheel
MAGMA_HOME="/usr/local/lib/magma2.6.1-cu101" MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ CMAKE_BUILD_TYPE=1 DEBUG=1 USE_LIBUV=1 USE_CUSPARSELT=1 USE_DISTRIBUTED=ON USE_MPI=ON USE_TENSORPIPE=ON USE_GLOO=ON USE_CUDA_MPI=ON python setup.py bdist_wheel # current running
MAGMA_HOME="/usr/local/lib/magma2.6.1-cu101" MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ USE_LIBUV=1 USE_CUSPARSELT=1 USE_DISTRIBUTED=ON USE_MPI=OFF USE_TENSORPIPE=ON USE_GLOO=ON USE_CUDA_MPI=ON python setup.py develop
```

## 1, Missing ATen cuda

```bash
/usr/local/bin/ccache /Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang++ -DHAVE_MMAP=1 -DHAVE_SHM_OPEN=1 -DHAVE_SHM_UNLINK=1 -DIDEEP_USE_MKL -DMINIZ_DISABLE_ZIP_READER_CRC32_CHECKS -DONNXIFI_ENABLE_EXT=1 -DONNX_ML=1 -DONNX_NAMESPACE=onnx_torch -DUSE_CUDA_MPI=1 -DUSE_EXTERNAL_MZCRC -D_FILE_OFFSET_BITS=64 -Dcaffe2_nvrtc_EXPORTS -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/build/aten/src -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/aten/src -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/build -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/cmake/../third_party/benchmark/include -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/third_party/onnx -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/build/third_party/onnx -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/third_party/foxi -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/build/third_party/foxi -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/build/third_party/gloo -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/cmake/../third_party/gloo -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/cmake/../third_party/tensorpipe/third_party/libuv/include -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/cmake/../third_party/googletest/googlemock/include -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/cmake/../third_party/googletest/googletest/include -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/third_party/protobuf/src -isystem /Users/llv23/opt/miniconda3/include -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/third_party/gemmlowp -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/third_party/neon2sse -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/third_party/XNNPACK/include -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/third_party/ittapi/include -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/cmake/../third_party/eigen -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/cmake/../third_party/cub -isystem /usr/local/include -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/third_party/ideep/include -D_LIBCPP_DISABLE_AVAILABILITY -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOCUPTI -DLIBKINETO_NOROCTRACER -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=braced-scalar-init -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wvla-extension -Wnewline-eof -Winconsistent-missing-override -Winconsistent-missing-destructor-override -Wno-pass-failed -Wno-error=pedantic -Wno-error=old-style-cast -Wno-error=inconsistent-missing-override -Wno-error=inconsistent-missing-destructor-override -Wconstant-conversion -Wno-invalid-partial-specialization -Wno-aligned-allocation-unavailable -Wno-missing-braces -Qunused-arguments -fcolor-diagnostics -faligned-new -fno-math-errno -fno-trapping-math -Werror=format -Wno-unused-private-field -Wno-missing-braces -DHAVE_AVX2_CPU_DEFINITION -O3 -DNDEBUG -DNDEBUG -std=gnu++14 -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.14.sdk -mmacosx-version-min=10.9 -fPIC -DMKL_HAS_SBGEMM -DTORCH_USE_LIBUV -DCAFFE2_USE_GLOO -MD -MT caffe2/CMakeFiles/caffe2_nvrtc.dir/__/aten/src/ATen/cuda/nvrtc_stub/ATenNVRTC.cpp.o -MF caffe2/CMakeFiles/caffe2_nvrtc.dir/__/aten/src/ATen/cuda/nvrtc_stub/ATenNVRTC.cpp.o.d -o caffe2/CMakeFiles/caffe2_nvrtc.dir/__/aten/src/ATen/cuda/nvrtc_stub/ATenNVRTC.cpp.o -c /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/aten/src/ATen/cuda/nvrtc_stub/ATenNVRTC.cpp
```

## 2, Migrating from c10 to std

```c++
# if defined(__APPLE__) && defined(__MACH__)
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
```

```c++
#if defined(__APPLE__) && defined(__MACH__)
#include <c10/util/Optional.h>
namespace std {
  using c10::optional;
}//namespace
#else
#include <optional>
#endif
```

```c++
#if defined(__APPLE__) && defined(__MACH__)
#include <c10/util/variant.h>
#endif
```

```c++
#if defined(__APPLE__) && defined(__MACH__)
#include <c10/util/variant.h>
#else
#include <variant>
#endif
```

```c++
#if defined(__APPLE__) && defined(__MACH__)
c10::visit
#else
#endif 
```

```c++
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
```

## 3, Issue of loading include headers

```bash
[1075/1631] Building CUDA object caffe2/CMakeFiles/torch_cuda.dir/__/aten/src/ATen/native/sparse/cuda/SparseSemiStructuredLinear.cu.o
FAILED: caffe2/CMakeFiles/torch_cuda.dir/__/aten/src/ATen/native/sparse/cuda/SparseSemiStructuredLinear.cu.o 
/usr/local/bin/ccache /usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler -ccbin=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang -DAT_PER_OPERATOR_HEADERS -DHAVE_MMAP=1 -DHAVE_SHM_OPEN=1 -DHAVE_SHM_UNLINK=1 -DIDEEP_USE_MKL -DMINIZ_DISABLE_ZIP_READER_CRC32_CHECKS -DONNXIFI_ENABLE_EXT=1 -DONNX_ML=1 -DONNX_NAMESPACE=onnx_torch -DTORCH_CUDA_BUILD_MAIN_LIB -DUSE_C10D_GLOO -DUSE_CUDA -DUSE_DISTRIBUTED -DUSE_EXPERIMENTAL_CUDNN_V8_API -DUSE_EXTERNAL_MZCRC -DUSE_RPC -DUSE_TENSORPIPE -D_FILE_OFFSET_BITS=64 -Dtorch_cuda_EXPORTS -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/build/aten/src -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/aten/src -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/build -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/cmake/../third_party/benchmark/include -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/third_party/onnx -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/build/third_party/onnx -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/third_party/foxi -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/build/third_party/foxi -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/c10/cuda/../.. -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/c10/.. -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/third_party/tensorpipe -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/cmake/../third_party/cutlass/include  -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/build/third_party/tensorpipe -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/third_party/tensorpipe/third_party/libnop/include -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/torch/csrc/api -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/torch/csrc/api/include -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/build/third_party/gloo -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/cmake/../third_party/gloo -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/cmake/../third_party/tensorpipe/third_party/libuv/include -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/cmake/../third_party/googletest/googlemock/include -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/cmake/../third_party/googletest/googletest/include -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/third_party/protobuf/src -isystem /Users/llv23/opt/miniconda3/include -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/third_party/gemmlowp -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/third_party/neon2sse -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/third_party/XNNPACK/include -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/third_party/ittapi/include -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/cmake/../third_party/eigen -isystem /usr/local/cuda/include -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/cmake/../third_party/cub -isystem /usr/local/include -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/third_party/ideep/include -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/cmake/../third_party/cudnn_frontend/include -isystem /usr/local/lib/magma2.6.1-cu101/include -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/cmake/../third_party/cutlass/include -Xfatbin -compress-all -DONNX_NAMESPACE=onnx_torch -gencode arch=compute_61,code=sm_61 -Xcudafe --diag_suppress=cc_clobber_ignored,--diag_suppress=field_without_dll_interface,--diag_suppress=base_class_has_different_dll_interface,--diag_suppress=dll_interface_conflict_none_assumed,--diag_suppress=dll_interface_conflict_dllexport_assumed,--diag_suppress=bad_friend_decl --expt-relaxed-constexpr --expt-extended-lambda  -Wno-deprecated-gpu-targets --expt-extended-lambda -DCUDA_HAS_FP16=1 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -g -std=c++14 -Xcompiler=-fPIC -DMKL_HAS_SBGEMM -DTORCH_USE_LIBUV -DCAFFE2_USE_GLOO -Xcompiler=-Wall,-Wextra,-Wdeprecated,-Wno-unused-parameter,-Wno-unused-function,-Wno-missing-field-initializers,-Wno-unknown-pragmas,-Wno-type-limits,-Wno-array-bounds,-Wno-unknown-pragmas,-Wno-strict-overflow,-Wno-strict-aliasing -MD -MT caffe2/CMakeFiles/torch_cuda.dir/__/aten/src/ATen/native/sparse/cuda/SparseSemiStructuredLinear.cu.o -MF caffe2/CMakeFiles/torch_cuda.dir/__/aten/src/ATen/native/sparse/cuda/SparseSemiStructuredLinear.cu.o.d -x cu -c /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/aten/src/ATen/native/sparse/cuda/SparseSemiStructuredLinear.cu -o caffe2/CMakeFiles/torch_cuda.dir/__/aten/src/ATen/native/sparse/cuda/SparseSemiStructuredLinear.cu.o
/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/aten/src/ATen/native/sparse/cuda/SparseSemiStructuredLinear.cu:8:10: fatal error: 'cutlass/cutlass.h' file not found
#include <cutlass/cutlass.h>
         ^~~~~~~~~~~~~~~~~~~
1 error generated.
[1076/1631] Building CUDA object caffe2/CMakeFiles/torch_cuda.dir/__/aten/src/ATen/native/nested/cuda/NestedTensorMatmul.cu.o
FAILED: caffe2/CMakeFiles/torch_cuda.dir/__/aten/src/ATen/native/nested/cuda/NestedTensorMatmul.cu.o 
/usr/local/bin/ccache /usr/local/cuda/bin/nvcc -forward-unknown-to-host-compiler -ccbin=/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/clang -DAT_PER_OPERATOR_HEADERS -DHAVE_MMAP=1 -DHAVE_SHM_OPEN=1 -DHAVE_SHM_UNLINK=1 -DIDEEP_USE_MKL -DMINIZ_DISABLE_ZIP_READER_CRC32_CHECKS -DONNXIFI_ENABLE_EXT=1 -DONNX_ML=1 -DONNX_NAMESPACE=onnx_torch -DTORCH_CUDA_BUILD_MAIN_LIB -DUSE_C10D_GLOO -DUSE_CUDA -DUSE_DISTRIBUTED -DUSE_EXPERIMENTAL_CUDNN_V8_API -DUSE_EXTERNAL_MZCRC -DUSE_RPC -DUSE_TENSORPIPE -D_FILE_OFFSET_BITS=64 -Dtorch_cuda_EXPORTS -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/build/aten/src -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/aten/src -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/build -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/cmake/../third_party/benchmark/include -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/third_party/onnx -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/build/third_party/onnx -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/third_party/foxi -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/build/third_party/foxi -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/c10/cuda/../.. -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/c10/.. -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/third_party/tensorpipe -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/build/third_party/tensorpipe -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/third_party/tensorpipe/third_party/libnop/include -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/torch/csrc/api -I/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/torch/csrc/api/include -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/build/third_party/gloo -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/cmake/../third_party/gloo -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/cmake/../third_party/tensorpipe/third_party/libuv/include -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/cmake/../third_party/googletest/googlemock/include -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/cmake/../third_party/googletest/googletest/include -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/third_party/protobuf/src -isystem /Users/llv23/opt/miniconda3/include -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/third_party/gemmlowp -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/third_party/neon2sse -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/third_party/XNNPACK/include -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/third_party/ittapi/include -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/cmake/../third_party/eigen -isystem /usr/local/cuda/include -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/cmake/../third_party/cub -isystem /usr/local/include -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/third_party/ideep/include -isystem /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/cmake/../third_party/cudnn_frontend/include -isystem /usr/local/lib/magma2.6.1-cu101/include -Xfatbin -compress-all -DONNX_NAMESPACE=onnx_torch -gencode arch=compute_61,code=sm_61 -Xcudafe --diag_suppress=cc_clobber_ignored,--diag_suppress=field_without_dll_interface,--diag_suppress=base_class_has_different_dll_interface,--diag_suppress=dll_interface_conflict_none_assumed,--diag_suppress=dll_interface_conflict_dllexport_assumed,--diag_suppress=bad_friend_decl --expt-relaxed-constexpr --expt-extended-lambda  -Wno-deprecated-gpu-targets --expt-extended-lambda -DCUDA_HAS_FP16=1 -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -g -std=c++14 -Xcompiler=-fPIC -DMKL_HAS_SBGEMM -DTORCH_USE_LIBUV -DCAFFE2_USE_GLOO -Xcompiler=-Wall,-Wextra,-Wdeprecated,-Wno-unused-parameter,-Wno-unused-function,-Wno-missing-field-initializers,-Wno-unknown-pragmas,-Wno-type-limits,-Wno-array-bounds,-Wno-unknown-pragmas,-Wno-strict-overflow,-Wno-strict-aliasing -MD -MT caffe2/CMakeFiles/torch_cuda.dir/__/aten/src/ATen/native/nested/cuda/NestedTensorMatmul.cu.o -MF caffe2/CMakeFiles/torch_cuda.dir/__/aten/src/ATen/native/nested/cuda/NestedTensorMatmul.cu.o.d -x cu -c /Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/aten/src/ATen/native/nested/cuda/NestedTensorMatmul.cu -o caffe2/CMakeFiles/torch_cuda.dir/__/aten/src/ATen/native/nested/cuda/NestedTensorMatmul.cu.o
/Users/llv23/Documents/05_machine_learning/dl_gpu_mac/pytorch-2.2.0-tensorpipe/aten/src/ATen/native/nested/cuda/NestedTensorMatmul.cu:22:10: fatal error: 'cutlass/gemm/device/default_gemm_configuration.h' file not found
#include <cutlass/gemm/device/default_gemm_configuration.h>
```

Solution: correct the caffe2/CMakeLists.txt in Line 96 and switch cutlass to 2.11.0, a prior version to 3.0.0 for CUDA 11.x

```cmake
 list(APPEND Caffe2_GPU_INCLUDE ${ATen_CUDA_INCLUDE} /usr/local/cuda/include ${PROJECT_SOURCE_DIR}/third_party/cutlass/include)
```

## 4. Runtime issue

torch 2.2.0's bash script result:

```bash
In [1]: import torch
libc++abi.dylib: terminating with uncaught exception of type std::runtime_error: arg(): could not convert default argument 'backend: c10::optional<c10::intrusive_ptr<c10d::Backend, c10::detail::intrusive_target_default_null_type<c10d::Backend> > >' in method '<class 'torch._C._distributed_c10d.ProcessGroup'>._register_backend' into a Python object (type not registered yet?)
Abort trap: 6
```

```bash
(base) Orlando:gpu-magma2.6.1-distributed-all-2.2.0-py3.10 llv23$ otool -L /Users/llv23/opt/miniconda3/lib/python3.10/site-packages/torch/lib/libtorch_python.dylib
/Users/llv23/opt/miniconda3/lib/python3.10/site-packages/torch/lib/libtorch_python.dylib:
	@rpath/libtorch_python.dylib (compatibility version 0.0.0, current version 0.0.0)
	@rpath/libshm.dylib (compatibility version 0.0.0, current version 0.0.0)
	@rpath/libtorch.dylib (compatibility version 0.0.0, current version 0.0.0)
	@rpath/libtorch_cuda.dylib (compatibility version 0.0.0, current version 0.0.0)
	@rpath/libnvToolsExt.1.dylib (compatibility version 0.0.0, current version 1.0.0)
	@rpath/libtorch_cpu.dylib (compatibility version 0.0.0, current version 0.0.0)
	@rpath/libmkl_intel_lp64.2.dylib (compatibility version 0.0.0, current version 0.0.0)
	@rpath/libmkl_intel_thread.2.dylib (compatibility version 0.0.0, current version 0.0.0)
	@rpath/libmkl_core.2.dylib (compatibility version 0.0.0, current version 0.0.0)
	@rpath/libomp.dylib (compatibility version 5.0.0, current version 5.0.0)
	/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1252.200.5)
	@rpath/libc10_cuda.dylib (compatibility version 0.0.0, current version 0.0.0)
	@rpath/libc10.dylib (compatibility version 0.0.0, current version 0.0.0)
	@rpath/libcudart.10.2.dylib (compatibility version 0.0.0, current version 10.2.89)
	@rpath/libcudnn.7.dylib (compatibility version 0.0.0, current version 7.6.5)
	/usr/local/opt/open-mpi/lib/libmpi.40.dylib (compatibility version 71.0.0, current version 71.1.0)
	/usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 400.9.4)
```

torch 2.0.0

```bash
(base) Orlando:lib llv23$ otool -L /Users/llv23/opt/miniconda3/lib/python3.10/site-packages/torch/lib/libtorch_python.dylib
/Users/llv23/opt/miniconda3/lib/python3.10/site-packages/torch/lib/libtorch_python.dylib:
	@rpath/libtorch_python.dylib (compatibility version 0.0.0, current version 0.0.0)
	@rpath/libshm.dylib (compatibility version 0.0.0, current version 0.0.0)
	/usr/local/opt/open-mpi/lib/libmpi.40.dylib (compatibility version 71.0.0, current version 71.1.0)
	@rpath/libtorch.dylib (compatibility version 0.0.0, current version 0.0.0)
	@rpath/libtorch_cuda.dylib (compatibility version 0.0.0, current version 0.0.0)
	@rpath/libnvrtc.10.1.dylib (compatibility version 0.0.0, current version 10.1.243)
	@rpath/libnvToolsExt.1.dylib (compatibility version 0.0.0, current version 1.0.0)
	@rpath/libtorch_cpu.dylib (compatibility version 0.0.0, current version 0.0.0)
	@rpath/libmkl_intel_lp64.2.dylib (compatibility version 0.0.0, current version 0.0.0)
	@rpath/libmkl_intel_thread.2.dylib (compatibility version 0.0.0, current version 0.0.0)
	@rpath/libmkl_core.2.dylib (compatibility version 0.0.0, current version 0.0.0)
	@rpath/libomp.dylib (compatibility version 5.0.0, current version 5.0.0)
	/usr/lib/libSystem.B.dylib (compatibility version 1.0.0, current version 1252.200.5)
	@rpath/libc10_cuda.dylib (compatibility version 0.0.0, current version 0.0.0)
	@rpath/libc10.dylib (compatibility version 0.0.0, current version 0.0.0)
	@rpath/libcudart.10.1.dylib (compatibility version 0.0.0, current version 10.1.243)
	@rpath/libcufft.10.dylib (compatibility version 0.0.0, current version 10.1.1)
	@rpath/libcurand.10.dylib (compatibility version 0.0.0, current version 10.1.1)
	@rpath/libcublas.10.dylib (compatibility version 0.0.0, current version 10.2.1)
	@rpath/libcublasLt.10.dylib (compatibility version 0.0.0, current version 10.2.1)
	@rpath/libcudnn.7.dylib (compatibility version 0.0.0, current version 7.6.5)
	/usr/lib/libc++.1.dylib (compatibility version 1.0.0, current version 400.9.4)
```

change torch/csrc/utils/pybind.h with 