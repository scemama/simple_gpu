#include <stdint.h>

/* =========================================================================
 * GPU Device Management
 * ========================================================================= */

/* Returns the number of available GPU devices on the system. */
int  gpu_ndevices();

/* Sets the active GPU device to the device at index i. */
void gpu_set_device(int32_t i);

/* Queries the memory status of the current GPU device.
 * Writes the free and total memory (in bytes) to the provided pointers. */
void gpu_get_memory(size_t* free, size_t* total);


/* =========================================================================
 * GPU Memory Management
 * ========================================================================= */

/* Allocates n bytes of memory on the GPU and writes the pointer to *ptr. */
void gpu_allocate(void** ptr, const int64_t n);

/* Frees GPU memory at *ptr and sets *ptr to NULL. */
void gpu_free(void** ptr);


/* =========================================================================
 * GPU Memory Transfer
 * ========================================================================= */

/* Copies n bytes from CPU memory (cpu_ptr) to GPU memory (gpu_ptr). */
void gpu_upload(const void* cpu_ptr, void* gpu_ptr, const int64_t n);

/* Copies n bytes from GPU memory (gpu_ptr) to CPU memory (cpu_ptr). */
void gpu_download(const void* gpu_ptr, void* cpu_ptr, const int64_t n);

/* Copies n bytes between two GPU memory regions (device-to-device). */
void gpu_copy(const void* gpu_ptr_src, void* gpu_ptr_dest, const int64_t n);


/* =========================================================================
 * GPU Stream Management
 * ========================================================================= */

/* Creates a GPU stream and writes its handle to *ptr.
 * Streams allow asynchronous, potentially concurrent kernel execution. */
void gpu_stream_create(void** ptr);

/* Destroys the GPU stream at *ptr and sets *ptr to NULL. */
void gpu_stream_destroy(void** ptr);

/* Associates a stream with a BLAS handle so that subsequent BLAS operations
 * are submitted to the given stream rather than the default stream. */
void gpu_set_stream(void* handle, void* stream);

/* Blocks the calling CPU thread until all pending GPU operations complete. */
void gpu_synchronize();


/* =========================================================================
 * GPU BLAS Handle Management
 * ========================================================================= */

/* Creates a GPU BLAS library handle and writes it to *handle.
 * The handle must be passed to all subsequent BLAS calls. */
void gpu_blas_create(void** handle);

/* Destroys a GPU BLAS library handle and sets *handle to NULL. */
void gpu_blas_destroy(void** handle);


/* =========================================================================
 * GPU BLAS Operations
 *
 * Conventions:
 *   - 'inc*' parameters are element strides (1 = contiguous).
 *   - 'ld*'  parameters are leading dimensions of matrices in memory.
 *   - 'trans' parameters control transposition: 'N' = no transpose,
 *             'T' = transpose, 'C' = conjugate transpose.
 *   - All matrix/vector pointers refer to GPU (device) memory.
 * ========================================================================= */

/* Double-precision dot product: result = x^T * y
 * Computes the inner product of two n-element vectors x and y. */
void gpu_ddot(const void* handle, const int64_t n,
              const double* x, const int64_t incx,
              const double* y, const int64_t incy,
              double* result);

/* Single-precision dot product: result = x^T * y
 * Computes the inner product of two n-element vectors x and y. */
void gpu_sdot(const void* handle, const int64_t n,
              const float* x, const int64_t incx,
              const float* y, const int64_t incy,
              float* result);

/* Double-precision matrix-vector multiply: y = alpha * op(A) * x + beta * y
 * op(A) is an m x n matrix (after applying transa), x is a vector of length n,
 * and y is a vector of length m. */
void gpu_dgemv(const void* handle, const char transa,
               const int64_t m, const int64_t n,
               const double* alpha,
               const double* a, const int64_t lda,
               const double* x, const int64_t incx,
               const double* beta, double* y, const int64_t incy);

/* Single-precision matrix-vector multiply: y = alpha * op(A) * x + beta * y
 * op(A) is an m x n matrix (after applying transa), x is a vector of length n,
 * and y is a vector of length m. */
void gpu_sgemv(const void* handle, const char transa,
               const int64_t m, const int64_t n,
               const float* alpha,
               const float* a, const int64_t lda,
               const float* x, const int64_t incx,
               const float* beta, float* y, const int64_t incy);

/* Double-precision matrix-matrix multiply: C = alpha * op(A) * op(B) + beta * C
 * op(A) is m x k, op(B) is k x n, and C is m x n. */
void gpu_dgemm(const void* handle, const char transa, const char transb,
               const int64_t m, const int64_t n, const int64_t k,
               const double* alpha,
               const double* a, const int64_t lda,
               const double* b, const int64_t ldb,
               const double* beta, double* c, const int64_t ldc);

/* Single-precision matrix-matrix multiply: C = alpha * op(A) * op(B) + beta * C
 * op(A) is m x k, op(B) is k x n, and C is m x n. */
void gpu_sgemm(const void* handle, const char transa, const char transb,
               const int64_t m, const int64_t n, const int64_t k,
               const float* alpha,
               const float* a, const int64_t lda,
               const float* b, const int64_t ldb,
               const float* beta, float* c, const int64_t ldc);

/* Double-precision matrix addition: C = alpha * op(A) + beta * op(B)
 * All matrices are m x n. op() applies the transposition specified by
 * transa and transb respectively. */
void gpu_dgeam(const void* handle, const char transa, const char transb,
               const int64_t m, const int64_t n,
               const double* alpha,
               const double* a, const int64_t lda,
               const double* beta,
               const double* b, const int64_t ldb,
               double* c, const int64_t ldc);

/* Single-precision matrix addition: C = alpha * op(A) + beta * op(B)
 * All matrices are m x n. op() applies the transposition specified by
 * transa and transb respectively. */
void gpu_sgeam(const void* handle, const char transa, const char transb,
               const int64_t m, const int64_t n,
               const float* alpha,
               const float* a, const int64_t lda,
               const float* beta,
               const float* b, const int64_t ldb,
               float* c, const int64_t ldc);
