!> @brief Simple GPU - Fortran GPU Computing Library
!>
!> This module provides a Fortran interface for GPU computing with transparent
!> support for both CPU (using standard BLAS) and NVIDIA GPU (using cuBLAS).
!>
!> @details
!> The library provides:
!> - GPU memory management (allocate, free, upload, download, copy)
!> - BLAS operations (Level 1, 2, and 3) for single and double precision
!> - Support for multidimensional arrays (1D through 6D)
!> - CUDA stream management for asynchronous operations
!> - Transparent API that works with both CPU and GPU backends
!>
!> @author Anthony Scemama
!> @date 2026
!>
module gpu
  use, intrinsic :: iso_c_binding
  implicit none

!=============================================================================
! Data Types
!=============================================================================
!
! GPU array types for double precision (1D through 6D)
! Each type contains:
!   - c: C pointer to GPU/CPU memory
!   - f: Fortran pointer for accessing data with Fortran array syntax
!
! Usage:
!   type(gpu_double1) :: x
!   call gpu_allocate(x, n)
!   x%f(i) = value  ! Access using Fortran syntax
!   call gpu_free(x)
!

  !> @brief 1-dimensional array of double precision values
  !> @details
  !> Use for vectors and 1D data. The Fortran pointer f(:) provides
  !> array-style access to the data.
  !> @code{.f90}
  !>   type(gpu_double1) :: x
  !>   call gpu_allocate(x, n)
  !>   x%f(1) = 1.0d0  ! Access first element
  !> @endcode
  type gpu_double1
    type(c_ptr) :: c                    !< C pointer to GPU/CPU memory
    double precision, pointer :: f(:)   !< Fortran pointer for array access
  end type

  !> @brief 2-dimensional array of double precision values
  !> @details
  !> Use for matrices and 2D data. The Fortran pointer f(:,:) provides
  !> array-style access to the data.
  !> @code{.f90}
  !>   type(gpu_double2) :: a
  !>   call gpu_allocate(a, m, n)
  !>   a%f(1,1) = 1.0d0  ! Access element at (1,1)
  !> @endcode
  type gpu_double2
    type(c_ptr) :: c                      !< C pointer to GPU/CPU memory
    double precision, pointer :: f(:,:)   !< Fortran pointer for array access
  end type

  !> @brief 3-dimensional array of double precision values
  type gpu_double3
    type(c_ptr) :: c                        !< C pointer to GPU/CPU memory
    double precision, pointer :: f(:,:,:)   !< Fortran pointer for array access
  end type

  !> @brief 4-dimensional array of double precision values
  type gpu_double4
    type(c_ptr) :: c                          !< C pointer to GPU/CPU memory
    double precision, pointer :: f(:,:,:,:)   !< Fortran pointer for array access
  end type

  !> @brief 5-dimensional array of double precision values
  type gpu_double5
    type(c_ptr) :: c                            !< C pointer to GPU/CPU memory
    double precision, pointer :: f(:,:,:,:,:)   !< Fortran pointer for array access
  end type

  !> @brief 6-dimensional array of double precision values
  type gpu_double6
    type(c_ptr) :: c                              !< C pointer to GPU/CPU memory
    double precision, pointer :: f(:,:,:,:,:,:)   !< Fortran pointer for array access
  end type

!
! GPU array types for single precision (1D through 6D)
!

  !> @brief 1-dimensional array of single precision values
  type gpu_real1
    type(c_ptr) :: c              !< C pointer to GPU/CPU memory
    real, pointer :: f(:)         !< Fortran pointer for array access
  end type

  !> @brief 2-dimensional array of single precision values
  type gpu_real2
    type(c_ptr) :: c                !< C pointer to GPU/CPU memory
    real, pointer :: f(:,:)         !< Fortran pointer for array access
  end type

  !> @brief 3-dimensional array of single precision values
  type gpu_real3
    type(c_ptr) :: c                  !< C pointer to GPU/CPU memory
    real, pointer :: f(:,:,:)         !< Fortran pointer for array access
  end type

  !> @brief 4-dimensional array of single precision values
  type gpu_real4
    type(c_ptr) :: c                    !< C pointer to GPU/CPU memory
    real, pointer :: f(:,:,:,:)         !< Fortran pointer for array access
  end type

  !> @brief 5-dimensional array of single precision values
  type gpu_real5
    type(c_ptr) :: c                      !< C pointer to GPU/CPU memory
    real, pointer :: f(:,:,:,:,:)         !< Fortran pointer for array access
  end type

  !> @brief 6-dimensional array of single precision values
  type gpu_real6
    type(c_ptr) :: c                        !< C pointer to GPU/CPU memory
    real, pointer :: f(:,:,:,:,:,:)         !< Fortran pointer for array access
  end type

!
! Handle types
!

  !> @brief Handle for BLAS operations
  !> @details
  !> Manages cuBLAS library context (for GPU) or BLAS context (for CPU).
  !> Must be created before calling any BLAS functions and destroyed when done.
  !> @code{.f90}
  !>   type(gpu_blas) :: handle
  !>   call gpu_blas_create(handle)
  !>   ! ... perform BLAS operations ...
  !>   call gpu_blas_destroy(handle)
  !> @endcode
  type gpu_blas
    type(c_ptr) :: c   !< C pointer to BLAS handle
  end type

  !> @brief Handle for CUDA streams
  !> @details
  !> Enables asynchronous operations and concurrent kernel execution.
  !> Only applicable when using the NVIDIA GPU backend.
  !> @code{.f90}
  !>   type(gpu_stream) :: stream
  !>   call gpu_stream_create(stream)
  !>   ! ... perform asynchronous operations ...
  !>   call gpu_stream_synchronize(stream)
  !>   call gpu_stream_destroy(stream)
  !> @endcode
  type gpu_stream
    type(c_ptr) :: c   !< C pointer to CUDA stream
  end type


!=============================================================================
! C interfaces
!=============================================================================
! Low-level C interface bindings - not intended for direct use
! These are wrapped by higher-level Fortran subroutines in the contains section
!

  interface
    !> @brief Get number of GPU devices (C binding)
    integer function gpu_ndevices() bind(C)
      import
    end function

    !> @brief Set active GPU device (C binding)
    subroutine gpu_set_device(id) bind(C)
      import
      integer(c_int32_t), value :: id
    end subroutine

    !> @brief Query GPU memory usage (C binding)
    subroutine gpu_get_memory(free, total) bind(C, name='gpu_get_memory')
      import
      integer(c_size_t) :: free, total
    end subroutine

    !> @brief Allocate GPU/CPU memory (C binding)
    subroutine gpu_allocate_c(ptr, n) bind(C, name='gpu_allocate')
      import
      type(c_ptr) :: ptr
      integer(c_int64_t), value :: n
    end subroutine

    !> @brief Free GPU/CPU memory (C binding)
    subroutine gpu_deallocate_c(ptr) bind(C, name='gpu_deallocate')
      import
      type(c_ptr) :: ptr
    end subroutine

    !> @brief Upload data to GPU (C binding)
    subroutine gpu_upload_c(cpu_ptr, gpu_ptr, n) bind(C, name='gpu_upload')
      import
      type(c_ptr), value :: cpu_ptr
      type(c_ptr), value :: gpu_ptr
      integer(c_int64_t), value :: n
    end subroutine

    subroutine gpu_download_c(gpu_ptr, cpu_ptr, n) bind(C, name='gpu_download')
      import
      type(c_ptr), value :: gpu_ptr
      type(c_ptr), value :: cpu_ptr
      integer(c_int64_t), value :: n
    end subroutine

    subroutine gpu_copy_c(gpu_ptr_src, gpu_ptr_dest, n) bind(C, name='gpu_copy')
      import
      type(c_ptr), value :: gpu_ptr_src
      type(c_ptr), value :: gpu_ptr_dest
      integer(c_int64_t), value :: n
    end subroutine

    subroutine gpu_stream_create_c(stream) bind(C, name='gpu_stream_create')
      import
      type(c_ptr) :: stream
    end subroutine

    subroutine gpu_stream_destroy_c(stream) bind(C, name='gpu_stream_destroy')
      import
      type(c_ptr) :: stream
    end subroutine

    subroutine gpu_set_stream_c(handle, stream) bind(C, name='gpu_set_stream')
      import
      type(c_ptr), value :: handle, stream
    end subroutine

    subroutine gpu_stream_synchronize(stream) bind(C)
      import
      type(c_ptr), value :: stream
    end subroutine

    subroutine gpu_synchronize() bind(C)
      import
    end subroutine

    subroutine gpu_blas_create_c(handle) bind(C, name='gpu_blas_create')
      import
      type(c_ptr) :: handle
    end subroutine

    subroutine gpu_blas_destroy_c(handle) bind(C, name='gpu_blas_destroy')
      import
      type(c_ptr) :: handle
    end subroutine

    subroutine gpu_ddot_c(handle, n, dx, incx, dy, incy, res) bind(C, name='gpu_ddot')
      import
      type(c_ptr), value, intent(in) :: handle
      integer(c_int64_t), value      :: n, incx, incy
      type(c_ptr), value             :: dx, dy
      real(c_double), intent(out)    :: res
    end subroutine

    subroutine gpu_sdot_c(handle, n, dx, incx, dy, incy, res) bind(C, name='gpu_sdot')
      import
      type(c_ptr), value, intent(in) :: handle
      integer(c_int64_t), value      :: n, incx, incy
      type(c_ptr), intent(in), value :: dx, dy
      real(c_float), intent(out)     :: res
    end subroutine

    subroutine gpu_dgeam_c(handle, transa, transb, m, n, alpha, a, lda, beta, &
      b, ldb, c, ldc) bind(C, name='gpu_dgeam')
      import
      type(c_ptr), value, intent(in)        :: handle
      character(c_char), intent(in)         :: transa, transb
      integer(c_int64_t), intent(in), value :: m, n, lda, ldb, ldc
      real(c_double), intent(in)            :: alpha, beta
      real(c_double) :: a, b, c
    end subroutine

    subroutine gpu_sgeam_c(handle, transa, transb, m, n, alpha, a, lda, beta, &
      b, ldb, c, ldc) bind(C, name='gpu_sgeam')
      import
      type(c_ptr), value, intent(in)        :: handle
      character(c_char), intent(in)         :: transa, transb
      integer(c_int64_t), intent(in), value :: m, n, lda, ldb, ldc
      real(c_float), intent(in)             :: alpha, beta
      real(c_float) :: a, b, c
    end subroutine

    subroutine gpu_dgemv_c(handle, transa, m, n, alpha, a, lda, &
      x, incx, beta, y, incy) bind(C, name='gpu_dgemv')
      import
      type(c_ptr), value, intent(in)        :: handle
      character(c_char), intent(in)         :: transa
      integer(c_int64_t), intent(in), value :: m, n, lda, incx, incy
      real(c_double), intent(in)            :: alpha, beta
      real(c_double)                        :: a, x, y
    end subroutine

    subroutine gpu_sgemv_c(handle, transa, m, n, alpha, a, lda, &
      x, incx, beta, y, incy) bind(C, name='gpu_sgemv')
      import
      type(c_ptr), value, intent(in)        :: handle
      character(c_char), intent(in)         :: transa
      integer(c_int64_t), intent(in), value :: m, n, lda, incx, incy
      real(c_float), intent(in)             :: alpha, beta
      real(c_float)                         :: a, x, y
    end subroutine


    subroutine gpu_dgemm_c(handle, transa, transb, m, n, k, alpha, a, lda, &
      b, ldb, beta, c, ldc) bind(C, name='gpu_dgemm')
      import
      type(c_ptr), value, intent(in)        :: handle
      character(c_char), intent(in)         :: transa, transb
      integer(c_int64_t), intent(in), value :: m, n, k, lda, ldb, ldc
      real(c_double), intent(in)            :: alpha, beta
      real(c_double) :: a, b, c
    end subroutine

    subroutine gpu_sgemm_c(handle, transa, transb, m, n, k, alpha, a, lda, &
      b, ldb, beta, c, ldc) bind(C, name='gpu_sgemm')
      import
      type(c_ptr), value, intent(in)        :: handle
      character(c_char), intent(in)         :: transa, transb
      integer(c_int64_t), intent(in), value :: m, n, k, lda, ldb, ldc
      real(c_float), intent(in)             :: alpha, beta
      real(c_float) :: a, b, c
    end subroutine

  end interface


!=============================================================================
! Polymorphic interfaces
!=============================================================================
!
! These interfaces are overloaded to work with different array types and
! dimensions. The appropriate implementation is selected automatically based
! on the argument types.
!

  !> @brief Allocate GPU/CPU memory for arrays
  !> @details
  !> Polymorphic interface that automatically selects the correct implementation
  !> based on the array type and number of dimensions provided.
  !>
  !> Supports:
  !> - 1D through 6D arrays
  !> - Single and double precision
  !> - 32-bit and 64-bit integer dimensions
  !>
  !> @param[out] arr GPU array to allocate (gpu_double1..6 or gpu_real1..6)
  !> @param[in] n1 Size of first dimension
  !> @param[in] n2 Size of second dimension (for 2D+)
  !> @param[in] n3 Size of third dimension (for 3D+)
  !> @param[in] n4 Size of fourth dimension (for 4D+)
  !> @param[in] n5 Size of fifth dimension (for 5D+)
  !> @param[in] n6 Size of sixth dimension (for 6D)
  !>
  !> @code{.f90}
  !>   type(gpu_double1) :: x
  !>   type(gpu_double2) :: a
  !>   call gpu_allocate(x, 1000)         ! 1D: 1000 elements
  !>   call gpu_allocate(a, 100, 200)     ! 2D: 100x200 matrix
  !> @endcode
  interface gpu_allocate
    procedure gpu_allocate_double1     &
             ,gpu_allocate_double2     &
             ,gpu_allocate_double3     &
             ,gpu_allocate_double4     &
             ,gpu_allocate_double5     &
             ,gpu_allocate_double6     &
             ,gpu_allocate_double1_64  &
             ,gpu_allocate_double2_64  &
             ,gpu_allocate_double3_64  &
             ,gpu_allocate_double4_64  &
             ,gpu_allocate_double5_64  &
             ,gpu_allocate_double6_64  &
             ,gpu_allocate_real1     &
             ,gpu_allocate_real2     &
             ,gpu_allocate_real3     &
             ,gpu_allocate_real4     &
             ,gpu_allocate_real5     &
             ,gpu_allocate_real6     &
             ,gpu_allocate_real1_64  &
             ,gpu_allocate_real2_64  &
             ,gpu_allocate_real3_64  &
             ,gpu_allocate_real4_64  &
             ,gpu_allocate_real5_64  &
             ,gpu_allocate_real6_64
  end interface gpu_allocate

  !> @brief Free GPU/CPU memory
  !> @details
  !> Deallocates memory previously allocated with gpu_allocate.
  !> After calling this, the array should not be used.
  !>
  !> @param[inout] arr GPU array to deallocate
  !>
  !> @code{.f90}
  !>   call gpu_deallocate(x)
  !> @endcode
  interface gpu_deallocate
    procedure gpu_deallocate_double1     &
             ,gpu_deallocate_double2     &
             ,gpu_deallocate_double3     &
             ,gpu_deallocate_double4     &
             ,gpu_deallocate_double5     &
             ,gpu_deallocate_double6     &
             ,gpu_deallocate_real1     &
             ,gpu_deallocate_real2     &
             ,gpu_deallocate_real3     &
             ,gpu_deallocate_real4     &
             ,gpu_deallocate_real5     &
             ,gpu_deallocate_real6
  end interface gpu_deallocate

  !> @brief Upload data from host (CPU) to device (GPU)
  !> @details
  !> Transfers data from host memory to GPU memory.
  !> The host array dimensions must match the GPU array dimensions.
  !>
  !> @param[in] host_arr Host array containing data to upload
  !> @param[inout] device_arr GPU array to receive data
  !>
  !> @code{.f90}
  !>   double precision :: x_h(1000)
  !>   type(gpu_double1) :: x
  !>   call gpu_allocate(x, 1000)
  !>   ! ... fill x_h with data ...
  !>   call gpu_upload(x_h, x)
  !> @endcode
  interface gpu_upload
    procedure gpu_upload_double0  &
             ,gpu_upload_double1  &
             ,gpu_upload_double2  &
             ,gpu_upload_double3  &
             ,gpu_upload_double4  &
             ,gpu_upload_double5  &
             ,gpu_upload_double6  &
             ,gpu_upload_real0  &
             ,gpu_upload_real1  &
             ,gpu_upload_real2  &
             ,gpu_upload_real3  &
             ,gpu_upload_real4  &
             ,gpu_upload_real5  &
             ,gpu_upload_real6
  end interface gpu_upload

  !> @brief Download data from device (GPU) to host (CPU)
  !> @details
  !> Transfers data from GPU memory to host memory.
  !> The host array dimensions must match the GPU array dimensions.
  !>
  !> @param[in] device_arr GPU array containing data to download
  !> @param[out] host_arr Host array to receive data
  !>
  !> @code{.f90}
  !>   double precision :: x_h(1000)
  !>   type(gpu_double1) :: x
  !>   call gpu_download(x, x_h)
  !> @endcode
  interface gpu_download
    procedure gpu_download_double0  &
             ,gpu_download_double1  &
             ,gpu_download_double2  &
             ,gpu_download_double3  &
             ,gpu_download_double4  &
             ,gpu_download_double5  &
             ,gpu_download_double6  &
             ,gpu_download_real0  &
             ,gpu_download_real1  &
             ,gpu_download_real2  &
             ,gpu_download_real3  &
             ,gpu_download_real4  &
             ,gpu_download_real5  &
             ,gpu_download_real6
  end interface gpu_download

  !> @brief Copy data between GPU arrays
  !> @details
  !> Copies data from one GPU array to another.
  !> Both arrays must have the same dimensions and type.
  !>
  !> @param[in] src Source GPU array
  !> @param[out] dest Destination GPU array
  !>
  !> @code{.f90}
  !>   type(gpu_double1) :: x, y
  !>   call gpu_allocate(x, 1000)
  !>   call gpu_allocate(y, 1000)
  !>   ! ... fill x with data ...
  !>   call gpu_copy(x, y)  ! Copy x to y
  !> @endcode
  interface gpu_copy
    procedure gpu_copy_double0  &
             ,gpu_copy_double1  &
             ,gpu_copy_double2  &
             ,gpu_copy_double3  &
             ,gpu_copy_double4  &
             ,gpu_copy_double5  &
             ,gpu_copy_double6  &
             ,gpu_copy_real0  &
             ,gpu_copy_real1  &
             ,gpu_copy_real2  &
             ,gpu_copy_real3  &
             ,gpu_copy_real4  &
             ,gpu_copy_real5  &
             ,gpu_copy_real6
  end interface gpu_copy


  contains

!=============================================================================
! Implementation of polymorphic interfaces
!=============================================================================

! gpu_allocate implementations
! ----------------------------
! These subroutines allocate GPU/CPU memory and set up Fortran pointers
! for convenient array access.

    !> @brief Allocate 1D double precision array (32-bit dimensions)
    subroutine gpu_allocate_double1(ptr, s)
      implicit none
      type(gpu_double1), intent(inout) :: ptr
      integer, intent(in) :: s
      integer*8 :: s_8, n

      s_8 = s
      n = s_8 * 8_8

      call gpu_allocate_c(ptr%c, n)
      call c_f_pointer(ptr%c, ptr%f, (/ s /))
    end subroutine

    subroutine gpu_allocate_double2(ptr, s1, s2)
      implicit none
      type(gpu_double2), intent(inout) :: ptr
      integer, intent(in) :: s1, s2
      integer*8 :: s1_8, s2_8, n

      s1_8 = s1
      s2_8 = s2
      n = s1_8 * s2_8 * 8_8

      call gpu_allocate_c(ptr%c, n)
      call c_f_pointer(ptr%c, ptr%f, (/ s1, s2 /))
    end subroutine

    subroutine gpu_allocate_double3(ptr, s1, s2, s3)
      implicit none
      type(gpu_double3), intent(inout) :: ptr
      integer, intent(in) :: s1, s2, s3
      integer*8 :: s1_8, s2_8, s3_8, n

      s1_8 = s1
      s2_8 = s2
      s3_8 = s3
      n = s1_8 * s2_8 * s3_8 * 8_8

      call gpu_allocate_c(ptr%c, n)
      call c_f_pointer(ptr%c, ptr%f, (/ s1, s2, s3 /))
    end subroutine

    subroutine gpu_allocate_double4(ptr, s1, s2, s3, s4)
      implicit none
      type(gpu_double4), intent(inout) :: ptr
      integer, intent(in) :: s1, s2, s3, s4
      integer*8 :: s1_8, s2_8, s3_8, s4_8, n

      s1_8 = s1
      s2_8 = s2
      s3_8 = s3
      s4_8 = s4
      n = s1_8 * s2_8 * s3_8 * s4_8 * 8_8

      call gpu_allocate_c(ptr%c, n)
      call c_f_pointer(ptr%c, ptr%f, (/ s1, s2, s3, s4 /))
    end subroutine

    subroutine gpu_allocate_double5(ptr, s1, s2, s3, s4, s5)
      implicit none
      type(gpu_double5), intent(inout) :: ptr
      integer, intent(in) :: s1, s2, s3, s4, s5
      integer*8 :: s1_8, s2_8, s3_8, s4_8, s5_8, n

      s1_8 = s1
      s2_8 = s2
      s3_8 = s3
      s4_8 = s4
      s5_8 = s5
      n = s1_8 * s2_8 * s3_8 * s4_8 * s5_8 * 8_8

      call gpu_allocate_c(ptr%c, n)
      call c_f_pointer(ptr%c, ptr%f, (/ s1, s2, s3, s4, s5 /))
    end subroutine

    subroutine gpu_allocate_double6(ptr, s1, s2, s3, s4, s5, s6)
      implicit none
      type(gpu_double6), intent(inout) :: ptr
      integer, intent(in) :: s1, s2, s3, s4, s5, s6
      integer*8 :: s1_8, s2_8, s3_8, s4_8, s5_8, s6_8, n

      s1_8 = s1
      s2_8 = s2
      s3_8 = s3
      s4_8 = s4
      s5_8 = s5
      s6_8 = s6
      n = s1_8 * s2_8 * s3_8 * s4_8 * s5_8 * s6_8 * 8_8

      call gpu_allocate_c(ptr%c, n)
      call c_f_pointer(ptr%c, ptr%f, (/ s1, s2, s3, s4, s5, s6 /))
    end subroutine


    subroutine gpu_allocate_double1_64(ptr, s)
      implicit none
      type(gpu_double1), intent(inout) :: ptr
      integer*8, intent(in) :: s

      call gpu_allocate_c(ptr%c, s)
      call c_f_pointer(ptr%c, ptr%f, (/ s /))
    end subroutine

    subroutine gpu_allocate_double2_64(ptr, s1, s2)
      implicit none
      type(gpu_double2), intent(inout) :: ptr
      integer*8, intent(in) :: s1, s2

      call gpu_allocate_c(ptr%c, s1*s2*8_8)
      call c_f_pointer(ptr%c, ptr%f, (/ s1, s2 /))
    end subroutine

    subroutine gpu_allocate_double3_64(ptr, s1, s2, s3)
      implicit none
      type(gpu_double3), intent(inout) :: ptr
      integer*8, intent(in) :: s1, s2, s3

      call gpu_allocate_c(ptr%c, s1*s2*s3*8_8)
      call c_f_pointer(ptr%c, ptr%f, (/ s1, s2, s3 /))
    end subroutine

    subroutine gpu_allocate_double4_64(ptr, s1, s2, s3, s4)
      implicit none
      type(gpu_double4), intent(inout) :: ptr
      integer*8, intent(in) :: s1, s2, s3, s4

      call gpu_allocate_c(ptr%c, s1*s2*s3*s4*8_8)
      call c_f_pointer(ptr%c, ptr%f, (/ s1, s2, s3, s4 /))
    end subroutine

    subroutine gpu_allocate_double5_64(ptr, s1, s2, s3, s4, s5)
      implicit none
      type(gpu_double5), intent(inout) :: ptr
      integer*8, intent(in) :: s1, s2, s3, s4, s5

      call gpu_allocate_c(ptr%c, s1*s2*s3*s4*s5*8_8)
      call c_f_pointer(ptr%c, ptr%f, (/ s1, s2, s3, s4, s5 /))
    end subroutine

    subroutine gpu_allocate_double6_64(ptr, s1, s2, s3, s4, s5, s6)
      implicit none
      type(gpu_double6), intent(inout) :: ptr
      integer*8, intent(in) :: s1, s2, s3, s4, s5, s6

      call gpu_allocate_c(ptr%c, s1*s2*s3*s4*s5*s6*8_8)
      call c_f_pointer(ptr%c, ptr%f, (/ s1, s2, s3, s4, s5, s6 /))
    end subroutine

    subroutine gpu_allocate_real1(ptr, s)
      implicit none
      type(gpu_real1), intent(inout) :: ptr
      integer, intent(in) :: s
      integer*8 :: s_8, n

      s_8 = s
      n = s_8 * 4_8

      call gpu_allocate_c(ptr%c, n)
      call c_f_pointer(ptr%c, ptr%f, (/ s /))
    end subroutine

    subroutine gpu_allocate_real2(ptr, s1, s2)
      implicit none
      type(gpu_real2), intent(inout) :: ptr
      integer, intent(in) :: s1, s2
      integer*8 :: s1_8, s2_8, n

      s1_8 = s1
      s2_8 = s2
      n = s1_8 * s2_8 * 4_8

      call gpu_allocate_c(ptr%c, n)
      call c_f_pointer(ptr%c, ptr%f, (/ s1, s2 /))
    end subroutine

    subroutine gpu_allocate_real3(ptr, s1, s2, s3)
      implicit none
      type(gpu_real3), intent(inout) :: ptr
      integer, intent(in) :: s1, s2, s3
      integer*8 :: s1_8, s2_8, s3_8, n

      s1_8 = s1
      s2_8 = s2
      s3_8 = s3
      n = s1_8 * s2_8 * s3_8 * 4_8

      call gpu_allocate_c(ptr%c, n)
      call c_f_pointer(ptr%c, ptr%f, (/ s1, s2, s3 /))
    end subroutine

    subroutine gpu_allocate_real4(ptr, s1, s2, s3, s4)
      implicit none
      type(gpu_real4), intent(inout) :: ptr
      integer, intent(in) :: s1, s2, s3, s4
      integer*8 :: s1_8, s2_8, s3_8, s4_8, n

      s1_8 = s1
      s2_8 = s2
      s3_8 = s3
      s4_8 = s4
      n = s1_8 * s2_8 * s3_8 * s4_8 * 4_8

      call gpu_allocate_c(ptr%c, n)
      call c_f_pointer(ptr%c, ptr%f, (/ s1, s2, s3, s4 /))
    end subroutine

    subroutine gpu_allocate_real5(ptr, s1, s2, s3, s4, s5)
      implicit none
      type(gpu_real5), intent(inout) :: ptr
      integer, intent(in) :: s1, s2, s3, s4, s5
      integer*8 :: s1_8, s2_8, s3_8, s4_8, s5_8, n

      s1_8 = s1
      s2_8 = s2
      s3_8 = s3
      s4_8 = s4
      s5_8 = s5
      n = s1_8 * s2_8 * s3_8 * s4_8 * s5_8 * 4_8

      call gpu_allocate_c(ptr%c, n)
      call c_f_pointer(ptr%c, ptr%f, (/ s1, s2, s3, s4, s5 /))
    end subroutine

    subroutine gpu_allocate_real6(ptr, s1, s2, s3, s4, s5, s6)
      implicit none
      type(gpu_real6), intent(inout) :: ptr
      integer, intent(in) :: s1, s2, s3, s4, s5, s6
      integer*8 :: s1_8, s2_8, s3_8, s4_8, s5_8, s6_8, n

      s1_8 = s1
      s2_8 = s2
      s3_8 = s3
      s4_8 = s4
      s5_8 = s5
      s6_8 = s6
      n = s1_8 * s2_8 * s3_8 * s4_8 * s5_8 * s6_8 * 4_8

      call gpu_allocate_c(ptr%c, n)
      call c_f_pointer(ptr%c, ptr%f, (/ s1, s2, s3, s4, s5, s6 /))
    end subroutine


    subroutine gpu_allocate_real1_64(ptr, s)
      implicit none
      type(gpu_real1), intent(inout) :: ptr
      integer*8, intent(in) :: s

      call gpu_allocate_c(ptr%c, s)
      call c_f_pointer(ptr%c, ptr%f, (/ s /))
    end subroutine

    subroutine gpu_allocate_real2_64(ptr, s1, s2)
      implicit none
      type(gpu_real2), intent(inout) :: ptr
      integer*8, intent(in) :: s1, s2

      call gpu_allocate_c(ptr%c, s1*s2*4_8)
      call c_f_pointer(ptr%c, ptr%f, (/ s1, s2 /))
    end subroutine

    subroutine gpu_allocate_real3_64(ptr, s1, s2, s3)
      implicit none
      type(gpu_real3), intent(inout) :: ptr
      integer*8, intent(in) :: s1, s2, s3

      call gpu_allocate_c(ptr%c, s1*s2*s3*4_8)
      call c_f_pointer(ptr%c, ptr%f, (/ s1, s2, s3 /))
    end subroutine

    subroutine gpu_allocate_real4_64(ptr, s1, s2, s3, s4)
      implicit none
      type(gpu_real4), intent(inout) :: ptr
      integer*8, intent(in) :: s1, s2, s3, s4

      call gpu_allocate_c(ptr%c, s1*s2*s3*s4*4_8)
      call c_f_pointer(ptr%c, ptr%f, (/ s1, s2, s3, s4 /))
    end subroutine

    subroutine gpu_allocate_real5_64(ptr, s1, s2, s3, s4, s5)
      implicit none
      type(gpu_real5), intent(inout) :: ptr
      integer*8, intent(in) :: s1, s2, s3, s4, s5

      call gpu_allocate_c(ptr%c, s1*s2*s3*s4*s5*4_8)
      call c_f_pointer(ptr%c, ptr%f, (/ s1, s2, s3, s4, s5 /))
    end subroutine

    subroutine gpu_allocate_real6_64(ptr, s1, s2, s3, s4, s5, s6)
      implicit none
      type(gpu_real6), intent(inout) :: ptr
      integer*8, intent(in) :: s1, s2, s3, s4, s5, s6

      call gpu_allocate_c(ptr%c, s1*s2*s3*s4*s5*s6*4_8)
      call c_f_pointer(ptr%c, ptr%f, (/ s1, s2, s3, s4, s5, s6 /))
    end subroutine

! gpu_deallocate
! --------------

    subroutine gpu_deallocate_double1(ptr)
      implicit none
      type(gpu_double1), intent(inout) :: ptr
      call gpu_deallocate_c(ptr%c)
      NULLIFY(ptr%f)
    end subroutine

    subroutine gpu_deallocate_double2(ptr)
      implicit none
      type(gpu_double2), intent(inout) :: ptr
      call gpu_deallocate_c(ptr%c)
      NULLIFY(ptr%f)
    end subroutine

    subroutine gpu_deallocate_double3(ptr)
      implicit none
      type(gpu_double3), intent(inout) :: ptr
      call gpu_deallocate_c(ptr%c)
      NULLIFY(ptr%f)
    end subroutine

    subroutine gpu_deallocate_double4(ptr)
      implicit none
      type(gpu_double4), intent(inout) :: ptr
      call gpu_deallocate_c(ptr%c)
      NULLIFY(ptr%f)
    end subroutine

    subroutine gpu_deallocate_double5(ptr)
      implicit none
      type(gpu_double5), intent(inout) :: ptr
      call gpu_deallocate_c(ptr%c)
      NULLIFY(ptr%f)
    end subroutine

    subroutine gpu_deallocate_double6(ptr)
      implicit none
      type(gpu_double6), intent(inout) :: ptr
      call gpu_deallocate_c(ptr%c)
      NULLIFY(ptr%f)
    end subroutine


    subroutine gpu_deallocate_real1(ptr)
      implicit none
      type(gpu_real1), intent(inout) :: ptr
      call gpu_deallocate_c(ptr%c)
      NULLIFY(ptr%f)
    end subroutine

    subroutine gpu_deallocate_real2(ptr)
      implicit none
      type(gpu_real2), intent(inout) :: ptr
      call gpu_deallocate_c(ptr%c)
      NULLIFY(ptr%f)
    end subroutine

    subroutine gpu_deallocate_real3(ptr)
      implicit none
      type(gpu_real3), intent(inout) :: ptr
      call gpu_deallocate_c(ptr%c)
      NULLIFY(ptr%f)
    end subroutine

    subroutine gpu_deallocate_real4(ptr)
      implicit none
      type(gpu_real4), intent(inout) :: ptr
      call gpu_deallocate_c(ptr%c)
      NULLIFY(ptr%f)
    end subroutine

    subroutine gpu_deallocate_real5(ptr)
      implicit none
      type(gpu_real5), intent(inout) :: ptr
      call gpu_deallocate_c(ptr%c)
      NULLIFY(ptr%f)
    end subroutine

    subroutine gpu_deallocate_real6(ptr)
      implicit none
      type(gpu_real6), intent(inout) :: ptr
      call gpu_deallocate_c(ptr%c)
      NULLIFY(ptr%f)
    end subroutine


! gpu_upload
! ----------

    subroutine gpu_upload_double0(cpu_ptr, gpu_ptr, n)
      implicit none
      double precision, target, intent(in)     :: cpu_ptr
      double precision, target, intent(in)     :: gpu_ptr
      integer, intent(in) :: n
      call gpu_upload_c(c_loc(cpu_ptr), c_loc(gpu_ptr), 8_8*n)
    end subroutine

    subroutine gpu_upload_double1(cpu_ptr, gpu_ptr)
      implicit none
      double precision, target, intent(in)     :: cpu_ptr(*)
      type(gpu_double1), intent(in)    :: gpu_ptr
      call gpu_upload_c(c_loc(cpu_ptr), gpu_ptr%c, 8_8*size(gpu_ptr%f))
    end subroutine

    subroutine gpu_upload_double2(cpu_ptr, gpu_ptr)
      implicit none
      double precision, target, intent(in)     :: cpu_ptr(:,:)
      type(gpu_double2), intent(in)    :: gpu_ptr
      call gpu_upload_c(c_loc(cpu_ptr), gpu_ptr%c, product(shape(gpu_ptr%f)*1_8)*8_8)
    end subroutine

    subroutine gpu_upload_double3(cpu_ptr, gpu_ptr)
      implicit none
      double precision, target, intent(in)     :: cpu_ptr(:,:,:)
      type(gpu_double3), intent(in)    :: gpu_ptr
      call gpu_upload_c(c_loc(cpu_ptr), gpu_ptr%c, product(shape(gpu_ptr%f)*1_8)*8_8)
    end subroutine

    subroutine gpu_upload_double4(cpu_ptr, gpu_ptr)
      implicit none
      double precision, target, intent(in)     :: cpu_ptr(:,:,:,:)
      type(gpu_double4), intent(in)    :: gpu_ptr
      call gpu_upload_c(c_loc(cpu_ptr), gpu_ptr%c, product(shape(gpu_ptr%f)*1_8)*8_8)
    end subroutine

    subroutine gpu_upload_double5(cpu_ptr, gpu_ptr)
      implicit none
      double precision, target, intent(in)     :: cpu_ptr(:,:,:,:,:)
      type(gpu_double5), intent(in)    :: gpu_ptr
      call gpu_upload_c(c_loc(cpu_ptr), gpu_ptr%c, product(shape(gpu_ptr%f)*1_8)*8_8)
    end subroutine

    subroutine gpu_upload_double6(cpu_ptr, gpu_ptr)
      implicit none
      double precision, target, intent(in)     :: cpu_ptr(:,:,:,:,:,:)
      type(gpu_double6), intent(in)    :: gpu_ptr
      call gpu_upload_c(c_loc(cpu_ptr), gpu_ptr%c, product(shape(gpu_ptr%f)*1_8)*8_8)
    end subroutine


    subroutine gpu_upload_real0(cpu_ptr, gpu_ptr, n)
      implicit none
      real, target, intent(in)     :: cpu_ptr
      real, target, intent(in)     :: gpu_ptr
      integer, intent(in) :: n
      call gpu_upload_c(c_loc(cpu_ptr), c_loc(gpu_ptr), 4_8*n)
    end subroutine

    subroutine gpu_upload_real1(cpu_ptr, gpu_ptr)
      implicit none
      real, target, intent(in)     :: cpu_ptr(*)
      type(gpu_real1), intent(in)    :: gpu_ptr
      call gpu_upload_c(c_loc(cpu_ptr), gpu_ptr%c, 4_8*size(gpu_ptr%f))
    end subroutine

    subroutine gpu_upload_real2(cpu_ptr, gpu_ptr)
      implicit none
      real, target, intent(in)     :: cpu_ptr(:,:)
      type(gpu_real2), intent(in)    :: gpu_ptr
      call gpu_upload_c(c_loc(cpu_ptr), gpu_ptr%c, product(shape(gpu_ptr%f)*1_8)*4_8)
    end subroutine

    subroutine gpu_upload_real3(cpu_ptr, gpu_ptr)
      implicit none
      real, target, intent(in)     :: cpu_ptr(:,:,:)
      type(gpu_real3), intent(in)    :: gpu_ptr
      call gpu_upload_c(c_loc(cpu_ptr), gpu_ptr%c, product(shape(gpu_ptr%f)*1_8)*4_8)
    end subroutine

    subroutine gpu_upload_real4(cpu_ptr, gpu_ptr)
      implicit none
      real, target, intent(in)     :: cpu_ptr(:,:,:,:)
      type(gpu_real4), intent(in)    :: gpu_ptr
      call gpu_upload_c(c_loc(cpu_ptr), gpu_ptr%c, product(shape(gpu_ptr%f)*1_8)*4_8)
    end subroutine

    subroutine gpu_upload_real5(cpu_ptr, gpu_ptr)
      implicit none
      real, target, intent(in)     :: cpu_ptr(:,:,:,:,:)
      type(gpu_real5), intent(in)    :: gpu_ptr
      call gpu_upload_c(c_loc(cpu_ptr), gpu_ptr%c, product(shape(gpu_ptr%f)*1_8)*4_8)
    end subroutine

    subroutine gpu_upload_real6(cpu_ptr, gpu_ptr)
      implicit none
      real, target, intent(in)     :: cpu_ptr(:,:,:,:,:,:)
      type(gpu_real6), intent(in)    :: gpu_ptr
      call gpu_upload_c(c_loc(cpu_ptr), gpu_ptr%c, product(shape(gpu_ptr%f)*1_8)*4_8)
    end subroutine


! gpu_download
! ------------

    subroutine gpu_download_double0(gpu_ptr, cpu_ptr, n)
      implicit none
      double precision, target, intent(in)     :: gpu_ptr
      double precision, target, intent(in)     :: cpu_ptr
      integer, intent(in) :: n
      call gpu_download_c(c_loc(gpu_ptr), c_loc(cpu_ptr), 8_8*n)
    end subroutine

    subroutine gpu_download_double1(gpu_ptr, cpu_ptr)
      implicit none
      type(gpu_double1), intent(in)  :: gpu_ptr
      double precision, target, intent(in)   :: cpu_ptr(:)
      call gpu_download_c(gpu_ptr%c, c_loc(cpu_ptr), 8_8*size(gpu_ptr%f))
    end subroutine

    subroutine gpu_download_double2(gpu_ptr, cpu_ptr)
      implicit none
      type(gpu_double2), intent(in)  :: gpu_ptr
      double precision, target, intent(in)   :: cpu_ptr(:,:)
      call gpu_download_c(gpu_ptr%c, c_loc(cpu_ptr), 8_8*product(shape(gpu_ptr%f)*1_8))
    end subroutine

    subroutine gpu_download_double3(gpu_ptr, cpu_ptr)
      implicit none
      type(gpu_double3), intent(in)  :: gpu_ptr
      double precision, target, intent(in)   :: cpu_ptr(:,:,:)
      call gpu_download_c(gpu_ptr%c, c_loc(cpu_ptr), 8_8*product(shape(gpu_ptr%f)*1_8))
    end subroutine

    subroutine gpu_download_double4(gpu_ptr, cpu_ptr)
      implicit none
      type(gpu_double4), intent(in)  :: gpu_ptr
      double precision, target, intent(in)   :: cpu_ptr(:,:,:,:)
      call gpu_download_c(gpu_ptr%c, c_loc(cpu_ptr), 8_8*product(shape(gpu_ptr%f)*1_8))
    end subroutine

    subroutine gpu_download_double5(gpu_ptr, cpu_ptr)
      implicit none
      type(gpu_double5), intent(in)  :: gpu_ptr
      double precision, target, intent(in)   :: cpu_ptr(:,:,:,:,:)
      call gpu_download_c(gpu_ptr%c, c_loc(cpu_ptr), 8_8*product(shape(gpu_ptr%f)*1_8))
    end subroutine

    subroutine gpu_download_double6(gpu_ptr, cpu_ptr)
      implicit none
      type(gpu_double6), intent(in)  :: gpu_ptr
      double precision, target, intent(in)   :: cpu_ptr(:,:,:,:,:,:)
      call gpu_download_c(gpu_ptr%c, c_loc(cpu_ptr), 8_8*product(shape(gpu_ptr%f)*1_8))
    end subroutine

    subroutine gpu_download_real0(gpu_ptr, cpu_ptr, n)
      implicit none
      real, target, intent(in)     :: gpu_ptr
      real, target, intent(in)     :: cpu_ptr
      integer, intent(in) :: n
      call gpu_download_c(c_loc(gpu_ptr), c_loc(cpu_ptr), 4_8*n)
    end subroutine

    subroutine gpu_download_real1(gpu_ptr, cpu_ptr)
      implicit none
      type(gpu_real1), intent(in)  :: gpu_ptr
      real, target, intent(in)   :: cpu_ptr(:)
      call gpu_download_c(gpu_ptr%c, c_loc(cpu_ptr), 4_8*size(gpu_ptr%f))
    end subroutine

    subroutine gpu_download_real2(gpu_ptr, cpu_ptr)
      implicit none
      type(gpu_real2), intent(in)  :: gpu_ptr
      real, target, intent(in)   :: cpu_ptr(:,:)
      call gpu_download_c(gpu_ptr%c, c_loc(cpu_ptr), 4_8*product(shape(gpu_ptr%f)*1_8))
    end subroutine

    subroutine gpu_download_real3(gpu_ptr, cpu_ptr)
      implicit none
      type(gpu_real3), intent(in)  :: gpu_ptr
      real, target, intent(in)   :: cpu_ptr(:,:,:)
      call gpu_download_c(gpu_ptr%c, c_loc(cpu_ptr), 4_8*product(shape(gpu_ptr%f)*1_8))
    end subroutine

    subroutine gpu_download_real4(gpu_ptr, cpu_ptr)
      implicit none
      type(gpu_real4), intent(in)  :: gpu_ptr
      real, target, intent(in)   :: cpu_ptr(:,:,:,:)
      call gpu_download_c(gpu_ptr%c, c_loc(cpu_ptr), 4_8*product(shape(gpu_ptr%f)*1_8))
    end subroutine

    subroutine gpu_download_real5(gpu_ptr, cpu_ptr)
      implicit none
      type(gpu_real5), intent(in)  :: gpu_ptr
      real, target, intent(in)   :: cpu_ptr(:,:,:,:,:)
      call gpu_download_c(gpu_ptr%c, c_loc(cpu_ptr), 4_8*product(shape(gpu_ptr%f)*1_8))
    end subroutine

    subroutine gpu_download_real6(gpu_ptr, cpu_ptr)
      implicit none
      type(gpu_real6), intent(in)  :: gpu_ptr
      real, target, intent(in)   :: cpu_ptr(:,:,:,:,:,:)
      call gpu_download_c(gpu_ptr%c, c_loc(cpu_ptr), 4_8*product(shape(gpu_ptr%f)*1_8))
    end subroutine

! gpu_copy
! --------

    subroutine gpu_copy_double0(gpu_ptr_src, gpu_ptr_dest, n)
      implicit none
      double precision, target, intent(in)        :: gpu_ptr_src
      double precision, target, intent(in)        :: gpu_ptr_dest
      integer, intent(in) :: n
      call gpu_copy_c(c_loc(gpu_ptr_src), c_loc(gpu_ptr_dest), 8_8*n)
    end subroutine

    subroutine gpu_copy_double1(gpu_ptr_src, gpu_ptr_dest)
      implicit none
      type(gpu_double1), intent(in)        :: gpu_ptr_src
      type(gpu_double1), intent(in)        :: gpu_ptr_dest
      call gpu_copy_c(gpu_ptr_src%c, gpu_ptr_dest%c, 8_8*size(gpu_ptr_dest%f))
    end subroutine

    subroutine gpu_copy_double2(gpu_ptr_src, gpu_ptr_dest)
      implicit none
      type(gpu_double2), intent(in)        :: gpu_ptr_src
      type(gpu_double2), intent(in)        :: gpu_ptr_dest
      call gpu_copy_c(gpu_ptr_src%c, gpu_ptr_dest%c, 8_8*product(shape(gpu_ptr_dest%f)*1_8))
    end subroutine

    subroutine gpu_copy_double3(gpu_ptr_src, gpu_ptr_dest)
      implicit none
      type(gpu_double3), intent(in)        :: gpu_ptr_src
      type(gpu_double3), intent(in)        :: gpu_ptr_dest
      call gpu_copy_c(gpu_ptr_src%c, gpu_ptr_dest%c, 8_8*product(shape(gpu_ptr_dest%f)*1_8))
    end subroutine

    subroutine gpu_copy_double4(gpu_ptr_src, gpu_ptr_dest)
      implicit none
      type(gpu_double4), intent(in)        :: gpu_ptr_src
      type(gpu_double4), intent(in)        :: gpu_ptr_dest
      call gpu_copy_c(gpu_ptr_src%c, gpu_ptr_dest%c, 8_8*product(shape(gpu_ptr_dest%f)*1_8))
    end subroutine

    subroutine gpu_copy_double5(gpu_ptr_src, gpu_ptr_dest)
      implicit none
      type(gpu_double5), intent(in)        :: gpu_ptr_src
      type(gpu_double5), intent(in)        :: gpu_ptr_dest
      call gpu_copy_c(gpu_ptr_src%c, gpu_ptr_dest%c, 8_8*product(shape(gpu_ptr_dest%f)*1_8))
    end subroutine

    subroutine gpu_copy_double6(gpu_ptr_src, gpu_ptr_dest)
      implicit none
      type(gpu_double6), intent(in)        :: gpu_ptr_src
      type(gpu_double6), intent(in)        :: gpu_ptr_dest
      call gpu_copy_c(gpu_ptr_src%c, gpu_ptr_dest%c, 8_8*product(shape(gpu_ptr_dest%f)*1_8))
    end subroutine

    subroutine gpu_copy_real0(gpu_ptr_src, gpu_ptr_dest, n)
      implicit none
      real, target, intent(in)        :: gpu_ptr_src
      real, target, intent(in)        :: gpu_ptr_dest
      integer, intent(in) :: n
      call gpu_copy_c(c_loc(gpu_ptr_src), c_loc(gpu_ptr_dest), 4_8*n)
    end subroutine

    subroutine gpu_copy_real1(gpu_ptr_src, gpu_ptr_dest)
      implicit none
      type(gpu_real1), intent(in)        :: gpu_ptr_src
      type(gpu_real1), intent(in)        :: gpu_ptr_dest
      call gpu_copy_c(gpu_ptr_src%c, gpu_ptr_dest%c, 4_8*size(gpu_ptr_dest%f))
    end subroutine

    subroutine gpu_copy_real2(gpu_ptr_src, gpu_ptr_dest)
      implicit none
      type(gpu_real2), intent(in)        :: gpu_ptr_src
      type(gpu_real2), intent(in)        :: gpu_ptr_dest
      call gpu_copy_c(gpu_ptr_src%c, gpu_ptr_dest%c, 4_8*product(shape(gpu_ptr_dest%f)*1_8))
    end subroutine

    subroutine gpu_copy_real3(gpu_ptr_src, gpu_ptr_dest)
      implicit none
      type(gpu_real3), intent(in)        :: gpu_ptr_src
      type(gpu_real3), intent(in)        :: gpu_ptr_dest
      call gpu_copy_c(gpu_ptr_src%c, gpu_ptr_dest%c, 4_8*product(shape(gpu_ptr_dest%f)*1_8))
    end subroutine

    subroutine gpu_copy_real4(gpu_ptr_src, gpu_ptr_dest)
      implicit none
      type(gpu_real4), intent(in)        :: gpu_ptr_src
      type(gpu_real4), intent(in)        :: gpu_ptr_dest
      call gpu_copy_c(gpu_ptr_src%c, gpu_ptr_dest%c, 4_8*product(shape(gpu_ptr_dest%f)*1_8))
    end subroutine

    subroutine gpu_copy_real5(gpu_ptr_src, gpu_ptr_dest)
      implicit none
      type(gpu_real5), intent(in)        :: gpu_ptr_src
      type(gpu_real5), intent(in)        :: gpu_ptr_dest
      call gpu_copy_c(gpu_ptr_src%c, gpu_ptr_dest%c, 4_8*product(shape(gpu_ptr_dest%f)*1_8))
    end subroutine

    subroutine gpu_copy_real6(gpu_ptr_src, gpu_ptr_dest)
      implicit none
      type(gpu_real6), intent(in)        :: gpu_ptr_src
      type(gpu_real6), intent(in)        :: gpu_ptr_dest
      call gpu_copy_c(gpu_ptr_src%c, gpu_ptr_dest%c, 4_8*product(shape(gpu_ptr_dest%f)*1_8))
    end subroutine


! gpu_stream
! ----------

    subroutine gpu_stream_create(stream)
      type(gpu_stream) :: stream
      call gpu_stream_create_c(stream%c)
    end subroutine

    subroutine gpu_stream_destroy(stream)
      type(gpu_stream) :: stream
      call gpu_stream_destroy_c(stream%c)
    end subroutine

    subroutine gpu_set_stream(handle, stream)
      type(gpu_blas)   :: handle
      type(gpu_stream) :: stream
      call gpu_set_stream_c(handle%c, stream%c)
    end subroutine


! gpu_blas
! BLAS handle management
! ----------------------

    !> @brief Create a BLAS handle
    !> @details
    !> Initializes a BLAS handle for performing BLAS operations.
    !> Must be called before any BLAS functions. The handle should be
    !> destroyed with gpu_blas_destroy when no longer needed.
    !>
    !> @param[out] handle BLAS handle to create
    subroutine gpu_blas_create(handle)
      type(gpu_blas) :: handle
      call gpu_blas_create_c(handle%c)
    end subroutine

    !> @brief Destroy a BLAS handle
    !> @details
    !> Frees resources associated with a BLAS handle.
    !> The handle should not be used after calling this.
    !>
    !> @param[inout] handle BLAS handle to destroy
    subroutine gpu_blas_destroy(handle)
      type(gpu_blas) :: handle
      call gpu_blas_destroy_c(handle%c)
    end subroutine




! BLAS Level 1: Vector operations
! --------------------------------

!> @brief Double precision dot product (32-bit dimensions)
!> @details
!> Computes the dot product of two vectors: result = x^T * y
!>
!> @param[in] handle BLAS handle
!> @param[in] n Number of elements
!> @param[in] dx First element of vector x
!> @param[in] incx Stride for vector x
!> @param[in] dy First element of vector y
!> @param[in] incy Stride for vector y
!> @param[out] res Resulting dot product
!>
!> @note Pass the first element of the array: x%f(1), not x
subroutine gpu_ddot(handle, n, dx, incx, dy, incy, res)
  type(gpu_blas), intent(in)     :: handle
  integer*4                      :: n, incx, incy
  double precision, target       :: dx, dy
  double precision, intent(out)  :: res
  call gpu_ddot_c(handle%c, int(n,c_int64_t), c_loc(dx), int(incx,c_int64_t), c_loc(dy), int(incy,c_int64_t), res)
end subroutine


!> @brief Double precision dot product (64-bit dimensions)
!> @details
!> Computes the dot product of two vectors: result = x^T * y
!> This variant accepts 64-bit integers for dimensions.
!>
!> @param[in] handle BLAS handle
!> @param[in] n Number of elements (64-bit)
!> @param[in] dx First element of vector x
!> @param[in] incx Stride for vector x (64-bit)
!> @param[in] dy First element of vector y
!> @param[in] incy Stride for vector y (64-bit)
!> @param[out] res Resulting dot product
subroutine gpu_ddot_64(handle, n, dx, incx, dy, incy, res)
  type(gpu_blas), intent(in)     :: handle
  integer*8                      :: n, incx, incy
  double precision, target       :: dx, dy
  double precision, intent(out)  :: res
  call gpu_ddot_c(handle%c, n, c_loc(dx), incx, c_loc(dy), incy, res)
end subroutine


subroutine gpu_sdot(handle, n, dx, incx, dy, incy, res)
  type(gpu_blas), intent(in)     :: handle
  integer*4                      :: n, incx, incy
  real, target       :: dx, dy
  real, intent(out)  :: res
  call gpu_sdot_c(handle%c, int(n,c_int64_t), c_loc(dx), int(incx,c_int64_t), c_loc(dy), int(incy,c_int64_t), res)
end subroutine


subroutine gpu_sdot_64(handle, n, dx, incx, dy, incy, res)
  type(gpu_blas), intent(in)     :: handle
  integer*8                      :: n, incx, incy
  real, target       :: dx, dy
  real, intent(out)  :: res
  call gpu_sdot_c(handle%c, n, c_loc(dx), incx, c_loc(dy), incy, res)
end subroutine


! geam
! ----

subroutine gpu_dgeam(handle, transa, transb, m, n, alpha, a, lda, beta, &
  b, ldb, c, ldc)
  type(gpu_blas), intent(in)   :: handle
  character, intent(in)        :: transa, transb
  integer*4, intent(in)        :: m, n, lda, ldb, ldc
  double precision, intent(in) :: alpha, beta
  double precision :: a, b, c
  call gpu_dgeam_c(handle%c, transa, transb, int(m,c_int64_t), int(n,c_int64_t), alpha, a, int(lda,c_int64_t), beta, &
        b, int(ldb,c_int64_t), c, int(ldc,c_int64_t))
end subroutine


subroutine gpu_dgeam_64(handle, transa, transb, m, n, alpha, a, lda, beta, &
  b, ldb, c, ldc)
  type(gpu_blas), intent(in)   :: handle
  character, intent(in)        :: transa, transb
  integer*8, intent(in)        :: m, n, lda, ldb, ldc
  double precision, intent(in) :: alpha, beta
  double precision      :: a, b, c
  call gpu_dgeam_c(handle%c, transa, transb, int(m,c_int64_t), int(n,c_int64_t), alpha, a, int(lda,c_int64_t), beta, &
        b, int(ldb,c_int64_t), c, int(ldc,c_int64_t))
end subroutine


subroutine gpu_sgeam(handle, transa, transb, m, n, alpha, a, lda, beta, &
  b, ldb, c, ldc)
  type(gpu_blas), intent(in)   :: handle
  character, intent(in)        :: transa, transb
  integer*4, intent(in)        :: m, n, lda, ldb, ldc
  real, intent(in) :: alpha, beta
  real :: a, b, c
  call gpu_sgeam_c(handle%c, transa, transb, int(m,c_int64_t), int(n,c_int64_t), alpha, a, int(lda,c_int64_t), beta, &
        b, int(ldb,c_int64_t), c, int(ldc,c_int64_t))
end subroutine


subroutine gpu_sgeam_64(handle, transa, transb, m, n, alpha, a, lda, beta, &
  b, ldb, c, ldc)
  type(gpu_blas), intent(in)   :: handle
  character, intent(in)        :: transa, transb
  integer*8, intent(in)        :: m, n, lda, ldb, ldc
  real, intent(in) :: alpha, beta
  real :: a, b, c
  call gpu_sgeam_c(handle%c, transa, transb, int(m,c_int64_t), int(n,c_int64_t), alpha, a, int(lda,c_int64_t), beta, &
        b, int(ldb,c_int64_t), c, int(ldc,c_int64_t))
end subroutine


! gemv
! ----

subroutine gpu_dgemv(handle, transa, m, n, alpha, a, lda, &
  x, incx, beta, y, incy)
  type(gpu_blas), intent(in)   :: handle
  character, intent(in)        :: transa
  integer*4, intent(in)        :: m, n, lda, incx, incy
  double precision, intent(in) :: alpha, beta
  double precision             :: a, x, y
  call gpu_dgemv_c(handle%c, transa, int(m,c_int64_t), int(n,c_int64_t), &
        alpha, a, int(lda,c_int64_t), &
        x, int(incx,c_int64_t), beta, y, int(incy,c_int64_t))
end subroutine

subroutine gpu_dgemv_64(handle, transa, m, n, alpha, a, lda, &
  x, incx, beta, y, incy)
  type(gpu_blas), intent(in)   :: handle
  character, intent(in)        :: transa
  integer*8, intent(in)        :: m, n, lda, incx, incy
  double precision, intent(in) :: alpha, beta
  double precision             :: a, x, y
  call gpu_dgemv_c(handle%c, transa, int(m,c_int64_t), int(n,c_int64_t), &
        alpha, a, int(lda,c_int64_t), &
        x, int(incx,c_int64_t), beta, y, int(incy,c_int64_t))
end subroutine


subroutine gpu_sgemv(handle, transa, m, n, alpha, a, lda, &
  x, incx, beta, y, incy)
  type(gpu_blas), intent(in)   :: handle
  character, intent(in)        :: transa
  integer*4, intent(in)        :: m, n, lda, incx, incy
  real, intent(in) :: alpha, beta
  real:: a, x, y
  call gpu_sgemv_c(handle%c, transa, int(m,c_int64_t), int(n,c_int64_t), &
        alpha, a, int(lda,c_int64_t), &
        x, int(incx,c_int64_t), beta, y, int(incy,c_int64_t))
end subroutine

subroutine gpu_sgemv_64(handle, transa, m, n, alpha, a, lda, &
  x, incx, beta, y, incy)
  type(gpu_blas), intent(in)   :: handle
  character, intent(in)        :: transa
  integer*8, intent(in)        :: m, n, lda, incx, incy
  real, intent(in) :: alpha, beta
  real:: a, x, y
  call gpu_sgemv_c(handle%c, transa, int(m,c_int64_t), int(n,c_int64_t), &
        alpha, a, int(lda,c_int64_t), &
        x, int(incx,c_int64_t), beta, y, int(incy,c_int64_t))
end subroutine


! gemm
! ----

subroutine gpu_dgemm(handle, transa, transb, m, n, k, alpha, a, lda, &
  b, ldb, beta, c, ldc)
  type(gpu_blas), intent(in)   :: handle
  character, intent(in)        :: transa, transb
  integer*4, intent(in)        :: m, n, k, lda, ldb, ldc
  double precision, intent(in) :: alpha, beta
  double precision             :: a, b, c
  call gpu_dgemm_c(handle%c, transa, transb, int(m,c_int64_t), int(n,c_int64_t), int(k,c_int64_t), &
        alpha, a, int(lda,c_int64_t), &
        b, int(ldb,c_int64_t), beta, c, int(ldc,c_int64_t))
end subroutine

subroutine gpu_dgemm_64(handle, transa, transb, m, n, k, alpha, a, lda, &
  b, ldb, beta, c, ldc)
  type(gpu_blas), intent(in)   :: handle
  character, intent(in)        :: transa, transb
  integer*8, intent(in)        :: m, n, k, lda, ldb, ldc
  double precision, intent(in) :: alpha, beta
  double precision             :: a, b, c
  call gpu_dgemm_c(handle%c, transa, transb, int(m,c_int64_t), int(n,c_int64_t), int(k,c_int64_t), &
        alpha, a, int(lda,c_int64_t), b, int(ldb,c_int64_t), beta, c, int(ldc,c_int64_t))
end subroutine

subroutine gpu_sgemm(handle, transa, transb, m, n, k, alpha, a, lda, &
  b, ldb, beta, c, ldc)
  type(gpu_blas), intent(in)   :: handle
  character, intent(in)        :: transa, transb
  integer*4, intent(in)        :: m, n, k, lda, ldb, ldc
  real, intent(in) :: alpha, beta
  real:: a, b, c
  call gpu_sgemm_c(handle%c, transa, transb, int(m,c_int64_t), int(n,c_int64_t), int(k,c_int64_t), &
        alpha, a, int(lda,c_int64_t), &
        b, int(ldb,c_int64_t), beta, c, int(ldc,c_int64_t))
end subroutine

subroutine gpu_sgemm_64(handle, transa, transb, m, n, k, alpha, a, lda, &
  b, ldb, beta, c, ldc)
  type(gpu_blas), intent(in)   :: handle
  character, intent(in)        :: transa, transb
  integer*8, intent(in)        :: m, n, k, lda, ldb, ldc
  real, intent(in) :: alpha, beta
  real:: a, b, c
  call gpu_sgemm_c(handle%c, transa, transb, int(m,c_int64_t), int(n,c_int64_t), int(k,c_int64_t), &
        alpha, a, int(lda,c_int64_t), b, int(ldb,c_int64_t), beta, c, int(ldc,c_int64_t))
end subroutine

end module
