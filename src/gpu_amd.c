#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include <hipblas/hipblas.h>
#include <hip/hip_runtime.h>


/* Generic functions */

int gpu_ndevices() {
  int ngpus;
  hipError_t rc = hipGetDeviceCount(&ngpus);
  if (rc != hipSuccess) {
    return 0;
  }
  return ngpus;
}

void gpu_set_device(int32_t igpu) {
 hipError_t rc = hipSetDevice((int)igpu);
 if (rc != hipSuccess) {
    fprintf(stderr,"hipSetDevice(%d) failed: %s\n", igpu, hipGetErrorString(rc));
    assert (rc == hipSuccess);
 }
}

void gpu_get_memory(size_t* free, size_t* total) {
    hipError_t rc = hipMemGetInfo( free, total );
    if (rc != hipSuccess) {
      *free = 0;
      *total = 0;
    }
}

/* Allocation functions */

void gpu_allocate(void** ptr, const int64_t size) {
    size_t free, total;
    hipError_t rc = hipMemGetInfo( &free, &total );
    if (rc != hipSuccess) {
      free = INT64_MAX;
    }

    rc = hipMalloc(ptr, size);

    if (rc != hipSuccess) {
      fprintf(stderr,"hipMalloc failed: %s\n", hipGetErrorString(rc));
      assert (rc == hipSuccess);
    }
}

void gpu_deallocate(void** ptr) {
  assert (*ptr != NULL);
  hipError_t rc = hipFree(*ptr);
  if (rc != hipSuccess) {
    fprintf(stderr,"hipFree failed: %s\n", hipGetErrorString(rc));
    assert (rc == hipSuccess);
  }
  *ptr = NULL;
}

void gpu_free(void** ptr) {
  gpu_deallocate(ptr);
}


/* Memory transfer functions */

void gpu_upload(const void* cpu_ptr, void* gpu_ptr, const int64_t n) {
 hipError_t rc = hipMemcpy (gpu_ptr, cpu_ptr, n, hipMemcpyHostToDevice);
 if (rc != hipSuccess) {
    fprintf(stderr,"hipMemcpy (upload) failed: %s\n", hipGetErrorString(rc));
    assert (rc == hipSuccess);
 }
}

void gpu_download(const void* gpu_ptr, void* cpu_ptr, const int64_t n) {
 hipError_t rc = hipMemcpy (cpu_ptr, gpu_ptr, n, hipMemcpyDeviceToHost);
 if (rc != hipSuccess) {
    fprintf(stderr,"hipMemcpy (download) failed: %s\n", hipGetErrorString(rc));
    assert (rc == hipSuccess);
 }
}

void gpu_copy(const void* gpu_ptr_src, void* gpu_ptr_dest, const int64_t n) {
 hipError_t rc = hipMemcpy (gpu_ptr_dest, gpu_ptr_src, n, hipMemcpyDeviceToDevice);
 if (rc != hipSuccess) {
   fprintf(stderr,"hipMemcpy (copy) failed: %s\n", hipGetErrorString(rc));
   assert (rc == hipSuccess);
 }
}


/* Streams */

void gpu_stream_create(hipStream_t* ptr) {
  hipError_t rc = hipStreamCreate(ptr);
  if (rc != hipSuccess) {
    fprintf(stderr,"hipStreamCreate failed: %s\n", hipGetErrorString(rc));
    assert (rc == hipSuccess);
  }
}

void gpu_stream_destroy(hipStream_t* ptr) {
  assert (ptr != NULL);
  hipError_t rc = hipStreamDestroy(*ptr);
  if (rc != hipSuccess) {
    fprintf(stderr,"hipStreamDestroy failed: %s\n", hipGetErrorString(rc));
    assert (rc == hipSuccess);
  }
  *ptr = NULL;
}

void gpu_set_stream(hipblasHandle_t handle, hipStream_t stream) {
  hipblasStatus_t rc = hipblasSetStream(handle, stream);
  if (rc != HIPBLAS_STATUS_SUCCESS) {
    fprintf(stderr,"hipblasSetStream failed\n");
    assert (rc == HIPBLAS_STATUS_SUCCESS);
  }
}

void gpu_synchronize() {
  hipError_t rc = hipDeviceSynchronize();
  if (rc != hipSuccess) {
    fprintf(stderr,"hipDeviceSynchronize failed: %s\n", hipGetErrorString(rc));
    assert (rc == hipSuccess);
  }
}

void gpu_stream_synchronize(void* stream) {
  hipError_t rc = hipStreamSynchronize(stream);
  if (rc != hipSuccess) {
    fprintf(stderr,"hipStreamSynchronize failed: %s\n", hipGetErrorString(rc));
    assert (rc == hipSuccess);
  }
}


/* BLAS functions */

void gpu_blas_create(hipblasHandle_t* ptr) {
  hipblasStatus_t rc = hipblasCreate(ptr);
  if (rc != HIPBLAS_STATUS_SUCCESS) {
    fprintf(stderr,"hipblasCreate failed\n");
  }
  assert (rc == HIPBLAS_STATUS_SUCCESS);
}


void gpu_blas_destroy(hipblasHandle_t* ptr) {
  assert (ptr != NULL);
  hipblasStatus_t rc = hipblasDestroy(*ptr);
  if (rc != HIPBLAS_STATUS_SUCCESS) {
    fprintf(stderr,"hipblasDestroy failed\n");
  }
  assert (rc == HIPBLAS_STATUS_SUCCESS);
  *ptr = NULL;
}


void gpu_ddot(hipblasHandle_t handle, const int64_t n, const double* x, const int64_t incx, const double* y, const int64_t incy, double* result) {
  assert (handle != NULL);
  /* Convert to int */
  int n_, incx_, incy_;

  n_    = (int)n;
  incx_ = (int)incx;
  incy_ = (int)incy;

  assert ( (int64_t)    n_ == n   );
  assert ( (int64_t) incx_ == incx);
  assert ( (int64_t) incy_ == incy);

  hipblasStatus_t rc = hipblasDdot(handle, n_, x, incx_, y, incy_, result);
  assert (rc == HIPBLAS_STATUS_SUCCESS);
}



void gpu_sdot(hipblasHandle_t handle, const int64_t n, const float* x, const int64_t incx, const float* y, const int64_t incy, float* result) {
  assert (handle != NULL);

  /* Convert to int */
  int n_, incx_, incy_;

  n_    = (int)n;
  incx_ = (int)incx;
  incy_ = (int)incy;

  /* Check for integer overflows */
  assert ( (int64_t)    n_ == n   );
  assert ( (int64_t) incx_ == incx);
  assert ( (int64_t) incy_ == incy);

  hipblasStatus_t rc = hipblasSdot(handle, n_, x, incx_, y, incy_, result);
  assert (rc == HIPBLAS_STATUS_SUCCESS);
}



void gpu_dgemv(hipblasHandle_t handle, const char* transa, const int64_t m, const int64_t n, const double* alpha,
               const double* a, const int64_t lda, const double* x, const int64_t incx, const double* beta, double* y, const int64_t incy) {

  assert (handle != NULL);

  /* Convert to int */
  int m_, n_, lda_, incx_, incy_;

  m_    = (int)m;
  n_    = (int)n;
  lda_  = (int)lda;
  incx_ = (int)incx;
  incy_ = (int)incy;

  /* Check for integer overflows */
  assert ( (int64_t)    m_ == m   );
  assert ( (int64_t)    n_ == n   );
  assert ( (int64_t)  lda_ == lda );
  assert ( (int64_t) incx_ == incx);
  assert ( (int64_t) incy_ == incy);

  hipblasOperation_t transa_ = HIPBLAS_OP_N;
  if (*transa == 'T' || *transa == 't') transa_ = HIPBLAS_OP_T;

  hipblasStatus_t rc = hipblasDgemv(handle, transa_, m_, n_, alpha, a, lda_, x, incx_, beta, y, incy_);
  assert (rc == HIPBLAS_STATUS_SUCCESS);
}



void gpu_sgemv(hipblasHandle_t handle, const char* transa, const int64_t m, const int64_t n, const float* alpha,
               const float* a, const int64_t lda, const float* x, const int64_t incx, const float* beta, float* y, const int64_t incy) {

  assert (handle != NULL);

  /* Convert to int */
  int m_, n_, lda_, incx_, incy_;

  m_    = (int)m;
  n_    = (int)n;
  lda_  = (int)lda;
  incx_ = (int)incx;
  incy_ = (int)incy;

  /* Check for integer overflows */
  assert ( (int64_t)    m_ == m   );
  assert ( (int64_t)    n_ == n   );
  assert ( (int64_t)  lda_ == lda );
  assert ( (int64_t) incx_ == incx);
  assert ( (int64_t) incy_ == incy);

  hipblasOperation_t transa_ = HIPBLAS_OP_N;
  if (*transa == 'T' || *transa == 't') transa_ = HIPBLAS_OP_T;

  hipblasStatus_t rc = hipblasSgemv(handle, transa_, m_, n_, alpha, a, lda_, x, incx_, beta, y, incy_);
  assert (rc == HIPBLAS_STATUS_SUCCESS);
}


void gpu_dgemm(hipblasHandle_t handle, const char* transa, const char* transb, const int64_t m, const int64_t n, const int64_t k, const double* alpha,
               const double* a, const int64_t lda, const double* b, const int64_t ldb, const double* beta, double* c, const int64_t ldc) {

  assert (handle != NULL);

  /* Convert to int */
  int m_, n_, k_, lda_, ldb_, ldc_;

  m_   = (int)m;
  n_   = (int)n;
  k_   = (int)k;
  lda_ = (int)lda;
  ldb_ = (int)ldb;
  ldc_ = (int)ldc;

  /* Check for integer overflows */
  assert ( (int64_t)   m_ == m  );
  assert ( (int64_t)   n_ == n  );
  assert ( (int64_t)   k_ == k  );
  assert ( (int64_t) lda_ == lda);
  assert ( (int64_t) ldb_ == ldb);
  assert ( (int64_t) ldc_ == ldc);

  hipblasOperation_t transa_ = HIPBLAS_OP_N;
  hipblasOperation_t transb_ = HIPBLAS_OP_N;
  if (*transa == 'T' || *transa == 't') transa_ = HIPBLAS_OP_T;
  if (*transb == 'T' || *transb == 't') transb_ = HIPBLAS_OP_T;

  hipblasStatus_t rc = hipblasDgemm(handle, transa_, transb_, m_, n_, k_, alpha, a, lda_, b, ldb_, beta, c, ldc_);
  assert (rc == HIPBLAS_STATUS_SUCCESS);
}



void gpu_sgemm(hipblasHandle_t handle, const char* transa, const char* transb, const int64_t m, const int64_t n, const int64_t k, const float* alpha,
               const float* a, const int64_t lda, const float* b, const int64_t ldb, const float* beta, float* c, const int64_t ldc) {

  assert (handle != NULL);

  /* Convert to int */
  int m_, n_, k_, lda_, ldb_, ldc_;

  m_   = (int)m;
  n_   = (int)n;
  k_   = (int)k;
  lda_ = (int)lda;
  ldb_ = (int)ldb;
  ldc_ = (int)ldc;

  /* Check for integer overflows */
  assert ( (int64_t)   m_ == m  );
  assert ( (int64_t)   n_ == n  );
  assert ( (int64_t)   k_ == k  );
  assert ( (int64_t) lda_ == lda);
  assert ( (int64_t) ldb_ == ldb);
  assert ( (int64_t) ldc_ == ldc);

  hipblasOperation_t transa_ = HIPBLAS_OP_N;
  hipblasOperation_t transb_ = HIPBLAS_OP_N;
  if (*transa == 'T' || *transa == 't') transa_ = HIPBLAS_OP_T;
  if (*transb == 'T' || *transb == 't') transb_ = HIPBLAS_OP_T;

  hipblasStatus_t rc = hipblasSgemm(handle, transa_, transb_, m_, n_, k_, alpha, a, lda_, b, ldb_, beta, c, ldc_);
  assert (rc == HIPBLAS_STATUS_SUCCESS);
}


void gpu_dgeam(hipblasHandle_t handle, const char* transa, const char* transb, const int64_t m, const int64_t n, const double* alpha,
               const double* a, const int64_t lda, const double* beta, const double* b, const int64_t ldb, double* c, const int64_t ldc) {
  assert (handle != NULL);

  /* Convert to int */
  int m_, n_, lda_, ldb_, ldc_;

  m_   = (int)m;
  n_   = (int)n;
  lda_ = (int)lda;
  ldb_ = (int)ldb;
  ldc_ = (int)ldc;

  /* Check for integer overflows */
  assert ( (int64_t)   m_ == m  );
  assert ( (int64_t)   n_ == n  );
  assert ( (int64_t) lda_ == lda);
  assert ( (int64_t) ldb_ == ldb);
  assert ( (int64_t) ldc_ == ldc);

  hipblasOperation_t transa_ = HIPBLAS_OP_N;
  hipblasOperation_t transb_ = HIPBLAS_OP_N;
  if (*transa == 'T' || *transa == 't') transa_ = HIPBLAS_OP_T;
  if (*transb == 'T' || *transb == 't') transb_ = HIPBLAS_OP_T;

  hipblasStatus_t rc = hipblasDgeam(handle, transa_, transb_, m_, n_, alpha, a, lda_, beta, b, ldb_, c, ldc_);
  assert (rc == HIPBLAS_STATUS_SUCCESS);

}


void gpu_sgeam(hipblasHandle_t handle, const char* transa, const char* transb, const int64_t m, const int64_t n, const float* alpha,
               const float* a, const int64_t lda, const float* beta, const float* b, const int64_t ldb, float* c, const int64_t ldc) {
  assert (handle != NULL);

  /* Convert to int */
  int m_, n_, lda_, ldb_, ldc_;

  m_   = (int)m;
  n_   = (int)n;
  lda_ = (int)lda;
  ldb_ = (int)ldb;
  ldc_ = (int)ldc;

  /* Check for integer overflows */
  assert ( (int64_t)   m_ == m  );
  assert ( (int64_t)   n_ == n  );
  assert ( (int64_t) lda_ == lda);
  assert ( (int64_t) ldb_ == ldb);
  assert ( (int64_t) ldc_ == ldc);

  hipblasOperation_t transa_ = HIPBLAS_OP_N;
  hipblasOperation_t transb_ = HIPBLAS_OP_N;
  if (*transa == 'T' || *transa == 't') transa_ = HIPBLAS_OP_T;
  if (*transb == 'T' || *transb == 't') transb_ = HIPBLAS_OP_T;

  hipblasStatus_t rc = hipblasSgeam(handle, transa_, transb_, m_, n_, alpha, a, lda_, beta, b, ldb_, c, ldc_);
  assert (rc == HIPBLAS_STATUS_SUCCESS);

}
