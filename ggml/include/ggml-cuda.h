#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// Get the primary CUDA stream of a CUDA backend. Returns NULL for non-CUDA backends.
// The returned stream can be used for issuing direct cudaMemcpyAsync calls that
// will be ordered correctly with subsequent compute ops on the same backend.
// Type is cudaStream_t; declared as void * to avoid forcing <cuda_runtime.h>
// into this header. Callers should cast: `(cudaStream_t)ggml_backend_cuda_get_stream(b)`.
GGML_BACKEND_API void * ggml_backend_cuda_get_stream(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(int main_device, const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

// Override the per-device op-offload min-batch threshold for MoE MUL_MAT_ID.
// CUDA's ggml_backend_cuda_device_offload_op() returns true only when
// get_op_batch_size(op) >= min_batch_size (default: 32 or GGML_OP_OFFLOAD_MIN_BATCH).
// For MUL_MAT_ID get_op_batch_size returns op->ne[2], which is 1 during
// single-token decode — so the scheduler keeps MoE on CPU and the
// per-expert host->device offload path (and any cache hooked into it)
// is never entered. Set this to 1 to force MoE MUL_MAT_ID onto GPU during
// decode. Safe to call after backend init. Returns false on non-CUDA build
// or invalid device.
GGML_BACKEND_API bool ggml_backend_cuda_set_op_offload_min_batch_size(int device, int min_batch_size);

#ifdef  __cplusplus
}
#endif
