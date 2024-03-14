<!-- markdownlint-disable MD010 -->
<!-- markdownlint-disable MD029 -->
# Migration note

Preparation of building library:

```bash
export CXXFLAGS=-D_LIBCPP_DISABLE_AVAILABILITY
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
MAGMA_HOME="/usr/local/lib/magma2.6.1-cu101" MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ USE_LIBUV=1 USE_DISTRIBUTED=ON USE_MPI=ON USE_TENSORPIPE=ON USE_GLOO=ON USE_CUDA_MPI=ON python setup.py clean
MAGMA_HOME="/usr/local/lib/magma2.6.1-cu101" MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ CMAKE_BUILD_TYPE=1 USE_LIBUV=1 USE_DISTRIBUTED=ON USE_MPI=ON USE_TENSORPIPE=ON USE_GLOO=ON USE_CUDA_MPI=ON python setup.py bdist_wheel
MAGMA_HOME="/usr/local/lib/magma2.6.1-cu101" MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ CMAKE_BUILD_TYPE=1 DEBUG=1 USE_LIBUV=1 USE_CUSPARSELT=1 USE_DISTRIBUTED=ON USE_MPI=ON USE_TENSORPIPE=ON USE_GLOO=ON USE_CUDA_MPI=ON BUILD_BUNDLE_PTXAS=1 python setup.py bdist_wheel
MAGMA_HOME="/usr/local/lib/magma2.6.1-cu101" MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ CMAKE_BUILD_TYPE=1 DEBUG=1 USE_LIBUV=1 USE_CUSPARSELT=1 USE_DISTRIBUTED=ON USE_MPI=ON USE_TENSORPIPE=ON USE_GLOO=ON USE_CUDA_MPI=ON python setup.py bdist_wheel # current running with removing libomp.dylib and libiomp5.dylib to /usr/local/lib
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

change torch/csrc/utils/pybind.h with cast_type.

## 5. Building pytorch.vision 0.17.1

Issue: not found  /usr/local/cuda/lib/libcudnn.a

Try with the following solution:

```bash
sudo ln -s  /usr/local/torch/lib/libdnnl.a /usr/local/lib/libdnnl.a
sudo ln -s  /usr/local/torch/lib/libc10_cuda.dylib /usr/local/lib/libc10_cuda.dylib
sudo ln -s  /usr/local/torch/lib/libc10.dylib /usr/local/lib/libc10.dylib
sudo ln -s  /usr/local/torch/lib/libtorch_cpu.dylib /usr/local/lib/libtorch_cpu.dylib
sudo ln -s  /usr/local/torch/lib/libtorch_cuda.dylib  /usr/local/lib/libtorch_cuda.dylib
sudo ln -s  /usr/local/torch/lib/libtorch.dylib  /usr/local/lib/libtorch.dylib
```

## 6, Upgrade from v2.2.0 to v2.2.1

1. CMakeLists.txt
2. cmake/Modules/FindMKLDNN.cmake
3. torch/__init__.py
4. torch/_dynamo/trace_rules.py
5. torch/_inductor/codecache.py
6. torch/_inductor/kernel/mm.py
7. torch/csrc/distributed/c10d/TCPStoreBackend.cpp
8. torch/csrc/lazy/core/shape_inference.cpp
9. torch/distributed/_shard/sharded_tensor/__init__.py
10. torch/distributed/_shard/sharded_tensor/api.py
11. torch/distributed/_tensor/__init__.py
12. torch/distributed/_tensor/ops/tensor_ops.py
13. torch/distributed/checkpoint/_state_dict_utils.py
14. torch/distributed/checkpoint/state_dict.py
15. torch/distributed/fsdp/_flat_param.py
16. torch/distributed/fsdp/_init_utils.py
17. torch/distributed/fsdp/_runtime_utils.py
18. torch/distributed/fsdp/fully_sharded_data_parallel.py
19. torch/distributed/tensor/parallel/_data_parallel_utils.py
20. torch/distributed/tensor/parallel/fsdp.py
21. torch/distributed/tensor/parallel/style.py
22. torch/fx/passes/split_module.py
23. torch/testing/_internal/common_dist_composable.py
24. docs/source/community/contribution_guide.rst
25. docs/source/nn.rst
26. test/distributed/_tensor/test_dtensor_compile.py
27. test/distributed/_tensor/test_tensor_ops.py
28. test/distributed/checkpoint/test_state_dict.py
29. test/distributed/fsdp/test_fsdp_freezing_weights.py
30. test/distributed/fsdp/test_fsdp_hybrid_shard.py
31. test/distributed/fsdp/test_fsdp_tp_integration.py
32. test/distributed/fsdp/test_hsdp_dtensor_state_dict.py
33. test/distributed/test_dynamo_distributed.py
34. test/lazy/test_meta_kernel.py

## 6, Decouple with local openmp - avoid link to /Users/llv23/opt/miniconda3/lib/libomp.dylib and /Users/llv23/opt/intel/oneapi//compiler/2021.4.0/mac/compiler/lib/libiomp5.dylib explicitly

For removing dependency with libraries /Users/llv23/opt/miniconda3/lib/libomp.dylib and /Users/llv23/opt/miniconda3/lib/libgomp.dylib.

1, remove compilation dependencies

* aten/src/ATen/CMakeLists.txt: Line 307 to Line 310

```bash
if(USE_OPENMP)
  message("ATen is compiled with OPEN_MP (/Users/llv23/opt/miniconda3/lib/libomp.dylib)")
  list(APPEND ATen_CPU_DEPENDENCY_LIBS /Users/llv23/opt/miniconda3/lib/libomp.dylib)
endif()
```

* caffe2/CMakeLists.txt Line: Line 102 to Line 103
* test/cpp/api/CMakeLists.txt Line 52 to Line 56 replacing with the following section  

```bash
list(APPEND Caffe2_CUDA_DEPENDENCY_LIBS ${ATen_CUDA_DEPENDENCY_LIBS})
```

2. Prepare libraries to /usr/local/include and /usr/local/lib:

/Users/llv23/opt/intel/oneapi//compiler/2021.4.0/mac/compiler/lib/libiomp5.dylib
/Users/llv23/opt/intel/oneapi//compiler/2021.4.0/mac/compiler/lib/libiomp5_db.dylib
/Users/llv23/opt/intel/oneapi//compiler/2021.4.0/mac/compiler/lib/libiompstubs5.dylib
/usr/local/Cellar/llvm/12.0.0_1//lib/libomp.dylib
/usr/local/Cellar/llvm/12.0.0_1//Toolchains/LLVM12.0.0.xctoolchain/usr/lib/libomp.dylib
/usr/local/Cellar/llvm/12.0.0_1//lib/clang/12.0.0/include/omp.h
/usr/local/Cellar/llvm/12.0.0_1//Toolchains/LLVM12.0.0.xctoolchain/usr/lib/clang/12.0.0/include/omp.h

We don't need "/Users/llv23/opt/miniconda3/lib/libomp.dylib -> /usr/local/Cellar//llvm/12.0.0_1/lib/libomp.dylib", if it has been compiled to /usr/local/include and /usr/local/lib.

```bash
rm -rf /usr/local/include/omp.h
rm -rf /usr/local/lib/libiomp5.dylib
rm -rf /usr/local/lib/libiomp5_db.dylib
rm -rf /usr/local/lib/libiompstubs5.dylib
rm -rf /usr/local/lib/libomp.dylib
# temporarily remove previously existing libraries
ln -s /usr/local/Cellar/llvm/12.0.0_1/lib/clang/12.0.0/include/omp.h /usr/local/include/omp.h
ln -s /Users/llv23/opt/intel/oneapi//compiler/2021.4.0/mac/compiler/lib/libiomp5.dylib /usr/local/lib/libiomp5.dylib
ln -s /Users/llv23/opt/intel/oneapi//compiler/2021.4.0/mac/compiler/lib/libiomp5_db.dylib /usr/local/lib/libiomp5_db.dylib
ln -s /Users/llv23/opt/intel/oneapi//compiler/2021.4.0/mac/compiler/lib/libiompstubs5.dylib /usr/local/lib/libiompstubs5.dylib
ln -s /usr/local/Cellar/llvm/12.0.0_1//Toolchains/LLVM12.0.0.xctoolchain/usr/lib/libomp.dylib /usr/local/lib/libomp.dylib
```
