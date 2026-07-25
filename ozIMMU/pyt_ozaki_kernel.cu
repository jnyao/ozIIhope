#include <torch/script.h>
#include "pyt_ozaki_kernel.hh"

#include <algorithm>
#include <chrono>
#include <cuComplex.h>
#include <cutf/curand.hpp>
#include <cutf/curand_kernel.hpp>
#include <cutf/device.hpp>
#include <cutf/math.hpp>
#include <cutf/memory.hpp>
#include <iostream>
#include <ozimmu/ozimmu.hpp>

#include <memory>
#include <string>
#include <vector>
#include <cmath>

#include <ctime>
#include <cfloat>
#include <iomanip>
#include <map>
#include <random>
#include <sstream>
#include <dlfcn.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math_constants.h>
#include <math.h>
#include <ATen/cudnn/Handle.h>


#define OZ_NORMAL 0          // 0
#define OZ_BSHALF 1

#define OZCUDNN_CHNLS 16
#define OZCUDNN_OUTCHNLS 64// 32 //64 // 72

inline void *ozIMMU_get_function_pointer(const std::string library_name,
                                         const std::string function_name) {

  // Open the library
  const auto lib_ptr = dlopen(library_name.c_str(), RTLD_NOW);
  if (lib_ptr == nullptr) {
    printf("Failed to load. Default rule will be used.");
    return nullptr;
  }

  // Get function pointer
  void *function_ptr = dlsym(lib_ptr, function_name.c_str());
  if (function_ptr == NULL) {
    printf(
        "Failed to load a function during selecting hijacking function. Default rule will be used.");
    return nullptr;
  }

  return function_ptr;
}


/** Error handling from https://developer.nvidia.com/cuDNN */
#define FatalError(s)                                                          \
  do {                                                                         \
    std::stringstream _where, _message;                                        \
    _where << __FILE__ << ':' << __LINE__;                                     \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;          \
    std::cerr << _message.str() << "\nAborting...\n";                          \
    cudaDeviceReset();                                                         \
    exit(1);                                                                   \
  } while (0)

#define checkCUDNN(status)                                                     \
  do {                                                                         \
    std::stringstream _error;                                                  \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);              \
      FatalError(_error.str());                                                \
    }                                                                          \
  } while (0)

#define checkCudaErrors(status)                                                \
  do {                                                                         \
    std::stringstream _error;                                                  \
    if (status != 0) {                                                         \
      _error << "Cuda failure: " << status;                                    \
      FatalError(_error.str());                                                \
    }                                                                          \
  } while (0)

// template <class T>
__global__ void init_accumulator_buffer_kernel(double *const dp_ptr,
                                               const std::size_t length) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= length) {
    return;
  }

  dp_ptr[tid] = 0;
}

// template <class T>
void init_accumulator_buffer(double *const dp_ptr, const std::size_t length,
                             cudaStream_t cuda_stream) {
  constexpr std::size_t block_size = 256;
  init_accumulator_buffer_kernel
      <<<(length + block_size - 1) / block_size, block_size, 0, cuda_stream>>>(
          dp_ptr, length);
}



template <class INPUT_T, class MANTISSA_T>
__device__ void cut_core_ozcudnn(at::Half *const out_ptr, float *const flt_ptr, const std::size_t inc1, const std::size_t inc2,
                              const INPUT_T a, const INPUT_T max_exp,
                              const unsigned num_split,
                              const unsigned mantissa_length, int tensortype) {
  const std::uint8_t sign_flag = a > 0;
  const std::uint64_t implict_one_bit =
      cutf::experimental::fp::mask_exponent(a) ? 1lu : 0lu;
  const auto mantissa =
      static_cast<MANTISSA_T>(
          cutf::experimental::fp::mask_mantissa(a) |
          (implict_one_bit
           << cutf::experimental::fp::get_mantissa_size<INPUT_T>()))
      << ((sizeof(MANTISSA_T) - sizeof(INPUT_T)) * 8 +
          cutf::experimental::fp::get_exponent_size<INPUT_T>());
  const auto mantissa_shift_offset =
      (cutf::experimental::fp::reinterpret_as_uint(max_exp) -
       cutf::experimental::fp::mask_exponent(a)) >>
      cutf::experimental::fp::get_mantissa_size<INPUT_T>();

  auto shifted_mantissa = mantissa >> mantissa_shift_offset;
  int curr_ind = 0;
    for (unsigned s = 0; s < num_split; s++) {
      const std::int32_t int8 =
          static_cast<std::int32_t>(shifted_mantissa >>
                                  (sizeof(MANTISSA_T) * 8 - mantissa_length)) *
          (sign_flag ? 1 : -1);
      shifted_mantissa <<= mantissa_length;

      if (tensortype == 0) {
        for (unsigned c = 0; c < num_split-s; c++) {
          out_ptr[s] = int8;
        }
      } else {
        for (unsigned c = 0; c < num_split-s; c++) {
          flt_ptr[(s+c) * inc1*OZCUDNN_CHNLS + (c) * inc2] = int8;
        }
      }
      curr_ind += (s+1);
    }
}

__global__ void split_ozcudnn(const double* x, size_t n, size_t inchnl, size_t outchnl, at::Half *const out_ptr,float *const flt_ptr,  const std::size_t inc, double max_exp, const unsigned num_split, const unsigned mantissa_length, int tensortype) {
  using MANTISSA_T = __uint128_t;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int inc_conv_chunk = inc / outchnl;
    if (idx < n) {
        double a = x[idx];
        if (tensortype == 0) {
          cut_core_ozcudnn<double, MANTISSA_T>(out_ptr + (idx) *OZCUDNN_CHNLS, flt_ptr, inc, inc/inchnl, a,
                                        max_exp, num_split, mantissa_length, tensortype);
        } else {
          cut_core_ozcudnn<double, MANTISSA_T>(out_ptr, flt_ptr + (idx / (inc_conv_chunk)) * inc_conv_chunk *OZCUDNN_CHNLS
                                        + idx % (inc_conv_chunk), inc, inc/outchnl, a,
                                        max_exp, num_split, mantissa_length, tensortype);
        }
    }
}


template <class INPUT_T, class MANTISSA_T>
__device__ void cut_core_ozcudnn_bs(at::Half *const out_ptr, float *const flt_ptr, const std::size_t inc1, const std::size_t inc2,
                              const INPUT_T a, const INPUT_T max_exp,
                              const unsigned num_split,
                              const unsigned mantissa_length, int tensortype) {
  const std::uint8_t sign_flag = a > 0;
  const std::uint64_t implict_one_bit =
      cutf::experimental::fp::mask_exponent(a) ? 1lu : 0lu;
  const auto mantissa =
      static_cast<MANTISSA_T>(
          cutf::experimental::fp::mask_mantissa(a) |
          (implict_one_bit
           << cutf::experimental::fp::get_mantissa_size<INPUT_T>()))
      << ((sizeof(MANTISSA_T) - sizeof(INPUT_T)) * 8 +
          cutf::experimental::fp::get_exponent_size<INPUT_T>());
  const auto mantissa_shift_offset =
      (cutf::experimental::fp::reinterpret_as_uint(max_exp) -
       cutf::experimental::fp::mask_exponent(a)) >>
      cutf::experimental::fp::get_mantissa_size<INPUT_T>();

  auto shifted_mantissa = mantissa >> mantissa_shift_offset;
  int curr_ind = 0;
    for (unsigned s = 0; s < num_split; s++) {
      const std::int32_t int8 =
          static_cast<std::int32_t>(shifted_mantissa >>
                                  (sizeof(MANTISSA_T) * 8 - mantissa_length)) *
          (sign_flag ? 1 : -1);
      shifted_mantissa <<= mantissa_length;

      if (tensortype == 0) {
        for (unsigned c = 0; c < num_split-s; c++) {
          out_ptr[s] = int8;
        }
      } else {
        for (unsigned c = 0; c < num_split-s; c++) {
          // int i = s+c;
          flt_ptr[(s+c) * inc1*OZCUDNN_CHNLS + (c) * inc2] = int8;
        }
      }
      curr_ind += (s+1);
    }
}

__global__ void split_ozcudnn_bs(const double* x, size_t n, size_t inchnl, size_t outchnl, at::Half *const out_ptr,float *const flt_ptr,  const std::size_t inc, double max_exp, const unsigned num_split, const unsigned mantissa_length, int tensortype) {
  using MANTISSA_T = __uint128_t;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int inc_conv_chunk = inc / outchnl;
    int batch_idx = idx % (inc/2);
    int batch_offset = idx / (inc/2);
    if (idx < n) {
        double a = x[idx];
        if (tensortype == 0) {
          cut_core_ozcudnn_bs<double, MANTISSA_T>(out_ptr + (batch_idx) *OZCUDNN_CHNLS + batch_offset * num_split, flt_ptr, inc, inc/inchnl, a,
                                        max_exp, num_split, mantissa_length, tensortype);
        } else {
          cut_core_ozcudnn_bs<double, MANTISSA_T>(out_ptr, flt_ptr + (idx / (inc_conv_chunk)) * inc_conv_chunk *OZCUDNN_CHNLS
                                        + idx % (inc_conv_chunk), inc, inc/outchnl, a,
                                        max_exp, num_split, mantissa_length, tensortype);
        }
    }
}

__global__ void accumulate_in_ozcudnn_bs_kernel(double *const f64_ptr,
                                         const float *i32_ptr,
                                         const std::size_t length,
                                         const int len_chunk,
                                         const std::int32_t outbs,
                                         const int numsplit,
                                         const int inumsplit,
                                         double max_exp) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= length) {
    return;
  }
  int batchidx = (tid / (length / 2));
  int tid32 = tid % (length / 2);
  int offset = (tid32/outbs) * (len_chunk-outbs) + batchidx * outbs * numsplit ;
  f64_ptr[tid] = 0;
  for (int i = 0; i < numsplit; i++) {
    int32_t a = i32_ptr[tid32 + offset + outbs*i];
    const auto scale = cutf::experimental::fp::reinterpret_as_fp(
      static_cast<std::uint64_t>(
          (cutf::experimental::fp::get_bias<double>() - inumsplit*i))
      << cutf::experimental::fp::get_mantissa_size<double>());
    f64_ptr[tid] +=
        static_cast<double>(static_cast<std::int64_t>(a) << 32) *scale / (1l << 44) * max_exp;
  }
}

void accumulate_in_ozcudnn_bs(double *const f64_ptr, const float *i32_ptr,
                       const std::size_t length,
                       const int len_chunk,
                       const std::int32_t outbs,
                       const int numsplit,
                       const int inumsplit,
                       double max_exp,
                       cudaStream_t cuda_stream) {
  constexpr std::size_t block_size = 256;
  accumulate_in_ozcudnn_bs_kernel<<<(length + block_size - 1) / block_size, block_size, 0, cuda_stream>>>(f64_ptr, i32_ptr, length, len_chunk, outbs, numsplit, inumsplit, max_exp);
}

__global__ void accumulate_in_ozcudnn_kernel(double *const f64_ptr,
                                         const float *i32_ptr,
                                         const std::size_t length,
                                         const int len_chunk,
                                         const std::int32_t outbs,
                                         const int numsplit,
                                         const int inumsplit,
                                         double max_exp) {
  const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= length) {
    return;
  }
  int offset = (tid/outbs) * (len_chunk-outbs);
  f64_ptr[tid] = 0;
  for (int i = 0; i < numsplit; i++) {
    int32_t a = i32_ptr[tid + offset + outbs*i];
    const auto scale = cutf::experimental::fp::reinterpret_as_fp(
      static_cast<std::uint64_t>(
          (cutf::experimental::fp::get_bias<double>() - inumsplit*i))
      << cutf::experimental::fp::get_mantissa_size<double>());
    f64_ptr[tid] +=
        static_cast<double>(static_cast<std::int64_t>(a) << 32) *scale / (1l << 44) * max_exp;
  }
}

void accumulate_in_ozcudnn(double *const f64_ptr, const float *i32_ptr,
                       const std::size_t length,
                       const int len_chunk,
                       const std::int32_t outbs,
                       const int numsplit,
                       const int inumsplit,
                       double max_exp,
                       cudaStream_t cuda_stream) {
  constexpr std::size_t block_size = 256;
  accumulate_in_ozcudnn_kernel<<<(length + block_size - 1) / block_size, block_size, 0, cuda_stream>>>(f64_ptr, i32_ptr, length, len_chunk, outbs, numsplit, inumsplit, max_exp);
}

/** Convolutional layer */
struct ConvolutionLayer_sg {
  int kernel_size;
  int in_channels, in_height, in_width;
  int out_channels, out_height, out_width;
  std::vector<float> pconv;

  ConvolutionLayer_sg(int in_channels_,
                   int out_channels_,
                   int kernel_size_,
                   int in_w_,
                   int in_h_)
    : pconv(in_channels_ * kernel_size_ * kernel_size_ * out_channels_) {
    in_channels = in_channels_;
    out_channels = out_channels_;
    kernel_size = kernel_size_;
    in_width = in_w_;
    in_height = in_h_;
    out_width = in_w_ - kernel_size_ + 1;
    out_height = in_h_ - kernel_size_ + 1;
  }
};


/** Split context */
struct SplitContext_oz {
  cudnnHandle_t cudnnHandle;
  cudnnTensorDescriptor_t dataTensor, conv1Tensor, conv1BiasTensor;
  cudnnFilterDescriptor_t conv1filterDesc;
  cudnnConvolutionDescriptor_t conv1Desc;
  cudnnConvolutionFwdAlgo_t conv1algo;
  int m_gpuid;
  int m_batchSize;
  size_t m_workspaceSize;
    // Create a CUDA stream
  cudaStream_t cuda_stream;
  float * d_data;
  float * d_conv1;
  float * d_pconv1;
  void * d_cudnn_workspace;

    // double* d_x;
    float* d_x1;
    float* d_x2;
    float* d_x3;
    float* d_x4;
    float* d_y1;
    std::int8_t* d_out_ptr;
    std::int8_t* d_yout_ptr;

  // Disable copying
  SplitContext_oz& operator=(const SplitContext_oz&) = delete;
  SplitContext_oz(const SplitContext_oz&) = delete;

  // Constructor
  SplitContext_oz(int gpuid, int batch_size, ConvolutionLayer_sg& conv1)
    : m_gpuid(gpuid) {
    m_batchSize = batch_size;

    checkCudaErrors(cudaSetDevice(gpuid));
    cudnnHandle = at::native::getCudnnHandle();
    cuda_stream = at::cuda::getCurrentCUDAStream();
  }

  ~SplitContext_oz() {
    checkCudaErrors(cudaSetDevice(m_gpuid));
  }

};


/** Split context */
struct SplitContext_ozcudnn {
  cudnnHandle_t cudnnHandle;
  cudnnTensorDescriptor_t dataTensor, conv1Tensor, conv1BiasTensor;
  cudnnFilterDescriptor_t conv1filterDesc;
  cudnnConvolutionDescriptor_t conv1Desc;
  cudnnConvolutionFwdAlgo_t conv1algo;
  int m_gpuid;
  int m_batchSize;
  size_t m_workspaceSize;
    // Create a CUDA stream
  cudaStream_t cuda_stream;
  float * d_data;
  float * d_conv1;
  float * d_pconv1;
  void * d_cudnn_workspace;

    // double* d_x;
    at::Half* d_x1;
    at::Half* d_x4;
    float* d_y1;
    std::int8_t* d_out_ptr;
    std::int8_t* d_yout_ptr;
  // int outbs;

  // Disable copying
  SplitContext_ozcudnn& operator=(const SplitContext_ozcudnn&) = delete;
  SplitContext_ozcudnn(const SplitContext_ozcudnn&) = delete;

  // Constructor
  SplitContext_ozcudnn(int gpuid, int batch_size, ConvolutionLayer_sg& conv1)
    : m_gpuid(gpuid) {
    m_batchSize = batch_size;

    checkCudaErrors(cudaSetDevice(gpuid));
    cudnnHandle = at::native::getCudnnHandle();
    cuda_stream = at::cuda::getCurrentCUDAStream();

  }

  ~SplitContext_ozcudnn() {
    checkCudaErrors(cudaSetDevice(m_gpuid));
  }

  /** Execute forward pass */
  void ForwardPropagation(
                          double* data,
                          double* conv1,
                          double* pconv1,
                          const std::size_t length,
                          const std::size_t n,
                          const std::size_t kn,
                          const std::size_t inchnl,
                          const std::size_t outchnl,
                          double max_exp) {
    const char* env_var = getenv("NUMSPLIT");
    int numsplit = atoi(env_var);
    env_var = getenv("BITS_PER_SLICE");
    int bits_per_slice = atoi(env_var);
    dim3 blockDim(256);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x);
    dim3 gridDimk((kn + blockDim.x - 1) / blockDim.x);

    split_ozcudnn<<<gridDim, blockDim>>>(data, n, inchnl, outchnl, d_x1, d_y1,  n, 2*(max_exp), numsplit, bits_per_slice, 0);
    split_ozcudnn<<<gridDim, blockDim>>>(pconv1, kn, inchnl, outchnl, d_x1, d_y1,  kn, 2*(max_exp), numsplit, bits_per_slice, 1);


  }
  /** Execute forward pass */
  void ForwardPropagation_bs(
                          double* data,
                          double* conv1,
                          double* pconv1,
                          const std::size_t length,
                          const std::size_t n,
                          const std::size_t kn,
                          const std::size_t inchnl,
                          const std::size_t outchnl,
                          double max_exp) {
    const char* env_var = getenv("NUMSPLIT");
    int numsplit = atoi(env_var);
    env_var = getenv("BITS_PER_SLICE");
    int bits_per_slice = atoi(env_var);
    dim3 blockDim(256);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x);
    dim3 gridDimk((kn + blockDim.x - 1) / blockDim.x);

    split_ozcudnn_bs<<<gridDim, blockDim>>>(data, n, inchnl, outchnl, d_x1, d_y1,  n, 2*(max_exp), numsplit, bits_per_slice, 0);
    split_ozcudnn_bs<<<gridDim, blockDim>>>(pconv1, kn, inchnl, outchnl, d_x1, d_y1,  kn, 2*(max_exp), numsplit, bits_per_slice, 1);

  }

};


torch::Tensor ozcudnn_launcher(torch::Tensor input1, torch::Tensor input2,torch::Tensor input3,torch::Tensor d_x,torch::Tensor d_y,torch::Tensor d_conv1){
    torch::Device device(torch::kCUDA, 0);
    int gpu = 0;//torch::cuda::current_device();

    // input dimensions
    size_t width = input1.size(3);
    size_t height = input1.size(2);
    size_t channels = input1.size(1);
    int batch_size = input1.size(0);

    // Create layer architecture
    int out_channels = input2.size(0);
    int kernel_size = input2.size(2);
    ConvolutionLayer_sg conv1_sg(
        (int)channels*OZCUDNN_CHNLS, out_channels, kernel_size, (int)width, (int)height);
    SplitContext_ozcudnn context_sg(gpu, batch_size, conv1_sg);

    context_sg.d_x1 = d_x.data_ptr<at::Half>();
    context_sg.d_y1 = d_y.data_ptr<float>();

    size_t n = batch_size * channels *
                                 height * width;
    size_t kn = conv1_sg.pconv.size() / OZCUDNN_CHNLS;
    const char* env_var = getenv("NUMSPLIT");
    int numsplit = atoi(env_var);

    void* d_cudnn_workspace_sg = nullptr;

        int bandwidth = *(d_conv1.data_ptr<int>());
    if (input1.device() == device){
        if (bandwidth == OZ_NORMAL) {
          context_sg.ForwardPropagation(input1.data_ptr<double>(), input3.data_ptr<double>(), input2.data_ptr<double>(),
                                    batch_size * conv1_sg.out_channels * conv1_sg.out_height * conv1_sg.out_width,
                                    n, kn, batch_size, conv1_sg.out_channels,0.5
                                    );

        }
        if (bandwidth == OZ_BSHALF) {
          context_sg.ForwardPropagation_bs(input1.data_ptr<double>(), input3.data_ptr<double>(), input2.data_ptr<double>(),
                                    batch_size * conv1_sg.out_channels * conv1_sg.out_height * conv1_sg.out_width,
                                    n, kn, batch_size, conv1_sg.out_channels,0.5
                                    );
        }
    }
    if (d_cudnn_workspace_sg != nullptr)
      checkCudaErrors(cudaFree(d_cudnn_workspace_sg));
    return input3;
}

torch::Tensor accumulate_ozcudnn_launcher(torch::Tensor input1, torch::Tensor input2,torch::Tensor input3,torch::Tensor d_x,torch::Tensor d_y,torch::Tensor d_conv1,torch::Tensor bw){
    torch::Device device(torch::kCUDA, 0);
    int gpu = 0;//torch::cuda::current_device();

    // input dimensions
    size_t width = input1.size(3);
    size_t height = input1.size(2);
    size_t channels = input1.size(1);
    int batch_size = input1.size(0);

    // Create layer architecture
    int out_channels = input2.size(0);
    int kernel_size = input2.size(2);
    ConvolutionLayer_sg conv1_sg(
        (int)channels*OZCUDNN_CHNLS, out_channels, kernel_size, (int)width, (int)height);
    SplitContext_oz context_sg(gpu, batch_size, conv1_sg);

    context_sg.d_conv1 = d_conv1.data_ptr<float>();
    context_sg.d_x1 = d_x.data_ptr<float>();
    context_sg.d_y1 = d_y.data_ptr<float>();

    size_t kn = conv1_sg.pconv.size() / OZCUDNN_CHNLS;
    const char* env_var = getenv("NUMSPLIT");
    int numsplit = atoi(env_var);

    void* d_cudnn_workspace_sg = nullptr;

    if (input1.device() == device){
        int bandwidth = *(bw.data_ptr<int>());
          const char* env_var = getenv("BITS_PER_SLICE");
          int bits_per_slice = atoi(env_var);
          const char* env_var_chl = getenv("OZCUDNN_OUTCHNLS");
          int outchnls = atoi(env_var_chl);
          if (bandwidth== OZ_NORMAL) {
            accumulate_in_ozcudnn(
              input3.data_ptr<double>(), d_conv1.data_ptr<float>(), batch_size * conv1_sg.out_channels * conv1_sg.out_height * conv1_sg.out_width,
              outchnls,
              conv1_sg.out_channels, 
              numsplit, 
              bits_per_slice,
              (2*0.5),
              context_sg.cuda_stream);
          } else if (bandwidth== OZ_BSHALF) {
            accumulate_in_ozcudnn_bs(
              input3.data_ptr<double>(), d_conv1.data_ptr<float>(), batch_size * conv1_sg.out_channels * conv1_sg.out_height * conv1_sg.out_width, 
              outchnls,
              conv1_sg.out_channels, 
              numsplit, 
              bits_per_slice,
              (2*0.5),
              context_sg.cuda_stream);
          }
    }
    if (d_cudnn_workspace_sg != nullptr)
      checkCudaErrors(cudaFree(d_cudnn_workspace_sg));
    return input3;
}
