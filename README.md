# Simple GPU - Fortran GPU Computing Library

A simple and easy-to-use library for GPU computing in Fortran, providing transparent access to GPU acceleration through a clean Fortran interface.

## Overview

Simple GPU is a library designed to simplify GPU computing in Fortran applications. It provides:

- **Dual implementation**: CPU-only version using standard BLAS, and GPU-accelerated version using NVIDIA cuBLAS
- **Transparent interface**: Same Fortran API for both CPU and GPU versions
- **Memory management**: Easy GPU memory allocation and data transfer
- **BLAS operations**: Common BLAS operations (GEMM, GEMV, DOT, GEAM) for both single and double precision
- **Stream support**: Asynchronous operations through CUDA streams

## Features

### Memory Management
- `gpu_allocate`: Allocate memory on GPU (or CPU for CPU version)
- `gpu_free`: Free allocated memory
- `gpu_upload`: Transfer data from CPU to GPU
- `gpu_download`: Transfer data from GPU to CPU
- `gpu_copy`: Copy data between GPU memory regions

### Device Management
- `gpu_ndevices`: Query number of available GPU devices
- `gpu_set_device`: Select active GPU device
- `gpu_get_memory`: Query GPU memory status

### BLAS Operations

All BLAS operations have variants that accept 64-bit integers for dimensions. These variants have a `_64` suffix (e.g., `gpu_ddot_64`, `gpu_dgemm_64`).

#### Level 1: Vector operations
- `gpu_sdot`, `gpu_ddot`: Dot product (single/double precision)

#### Level 2: Matrix-vector operations
- `gpu_sgemv`, `gpu_dgemv`: Matrix-vector multiplication

#### Level 3: Matrix-matrix operations
- `gpu_sgemm`, `gpu_dgemm`: Matrix-matrix multiplication
- `gpu_sgeam`, `gpu_dgeam`: Matrix addition/transposition

### Advanced Features
- **Streams**: Create and manage CUDA streams for asynchronous execution
- **BLAS handles**: Manage cuBLAS library handles
- **Stream synchronization**: Control execution flow

## Installation

### Prerequisites

#### For CPU version (required):
- C compiler (gcc, clang, etc.)
- Fortran compiler (gfortran, ifort, etc.)
- BLAS library (OpenBLAS, Intel MKL, or reference BLAS)
- Autotools (autoconf, automake, libtool)

#### For NVIDIA GPU version (optional):
- NVIDIA CUDA Toolkit (with nvcc compiler)
- NVIDIA cuBLAS library
- CUDA-capable GPU

### Building from Source

1. **Generate configure script** (if building from git):
   ```bash
   ./autogen.sh
   ```

2. **Configure the build**:
   ```bash
   ./configure
   ```
   
   The configure script will automatically detect if CUDA is available and enable the NVIDIA GPU library if possible.

   **Configuration options**:
   - `--disable-nvidia`: Disable NVIDIA GPU library even if CUDA is available
   - `--with-cuda=DIR`: Specify CUDA installation directory (default: `/usr/local/cuda`)
   - `--with-blas=LIB`: Specify BLAS library to use
   
   **Example configurations**:
   ```bash
   # CPU version only
   ./configure --disable-nvidia
   
   # Specify CUDA location
   ./configure --with-cuda=/opt/cuda
   
   # Use specific BLAS library
   ./configure --with-blas="-lmkl_rt"
   ```

3. **Build the libraries**:
   ```bash
   make
   ```

4. **Run tests** (optional):
   ```bash
   make check
   ```

5. **Install**:
   ```bash
   sudo make install
   ```

## Library Versions

Simple GPU provides two shared libraries:

1. **libgpu_cpu.so**: CPU-only version
   - Uses standard BLAS library
   - No GPU required
   - Useful for development and testing on systems without GPUs

2. **libgpu_nvidia.so**: NVIDIA GPU version (if CUDA is available)
   - Uses NVIDIA cuBLAS library
   - Requires CUDA-capable GPU
   - Provides GPU acceleration for supported operations

Both libraries provide the same Fortran interface, allowing seamless switching between CPU and GPU implementations.

## Usage

### Data Types

The library provides multidimensional array types for both single and double precision:

- `gpu_double1`: 1-dimensional array of double precision values
- `gpu_double2`: 2-dimensional array of double precision values
- `gpu_double3`: 3-dimensional array of double precision values
- `gpu_double4`, `gpu_double5`, `gpu_double6`: 4, 5, and 6-dimensional arrays

Similarly for single precision:
- `gpu_real1` through `gpu_real6`

Each type contains:
- `c`: C pointer to GPU memory
- `f`: Fortran pointer for accessing data (e.g., `f(:)` for 1D, `f(:,:)` for 2D)

The `gpu_allocate` function is overloaded and automatically accepts the appropriate number of dimensions:
```f90
call gpu_allocate(x, n)        ! 1D array: x is gpu_double1 or gpu_real1
call gpu_allocate(a, m, n)     ! 2D array: a is gpu_double2 or gpu_real2
call gpu_allocate(b, l, m, n)  ! 3D array: b is gpu_double3 or gpu_real3
```

### Basic Example with 1D Arrays

```f90
program example
  use gpu
  implicit none
  
  type(gpu_blas) :: handle
  type(gpu_double1) :: x, y
  double precision, allocatable :: x_h(:), y_h(:)
  double precision :: result
  integer :: n, i
  
  n = 1000
  
  ! Initialize BLAS handle
  call gpu_blas_create(handle)
  
  ! Allocate vectors (1D arrays)
  call gpu_allocate(x, n)
  call gpu_allocate(y, n)
  
  ! Create and initialize host data
  allocate(x_h(n), y_h(n))
  do i = 1, n
    x_h(i) = dble(i)
    y_h(i) = dble(i) * 2.0d0
  end do
  
  ! Upload to GPU
  call gpu_upload(x_h, x)
  call gpu_upload(y_h, y)
  
  ! Compute dot product on GPU
  ! Note: Use address of first element (x%f(1), not x)
  call gpu_ddot(handle, n, x%f(1), 1, y%f(1), 1, result)
  
  print *, 'Dot product result:', result
  
  ! Clean up
  deallocate(x_h, y_h)
  call gpu_free(x)
  call gpu_free(y)
  call gpu_blas_destroy(handle)
  
end program example
```

### Example with 2D Arrays

```f90
program example_2d
  use gpu
  implicit none
  
  type(gpu_blas) :: handle
  type(gpu_double2) :: a, b, c
  double precision, allocatable :: a_h(:,:), b_h(:,:), c_h(:,:)
  double precision :: alpha, beta
  integer :: m, n, i, j
  
  m = 100
  n = 200
  
  ! Initialize BLAS handle
  call gpu_blas_create(handle)
  
  ! Allocate matrices (2D arrays)
  call gpu_allocate(a, m, n)
  call gpu_allocate(b, m, n)
  call gpu_allocate(c, m, n)
  
  ! Create and initialize host data
  allocate(a_h(m,n), b_h(m,n), c_h(m,n))
  do j = 1, n
    do i = 1, m
      a_h(i,j) = dble(i + j)
      b_h(i,j) = dble(i * j)
    end do
  end do
  
  ! Upload to GPU
  call gpu_upload(a_h, a)
  call gpu_upload(b_h, b)
  
  ! Matrix addition: C = alpha*A + beta*B
  alpha = 1.5d0
  beta = 0.5d0
  ! Note: Use address of first element (a%f(1,1), not a)
  call gpu_dgeam(handle, 'N', 'N', m, n, &
                 alpha, a%f(1,1), m, beta, b%f(1,1), m, &
                 c%f(1,1), m)
  
  ! Download result from GPU to host
  call gpu_download(c, c_h)
  
  ! Access result in c_h(:,:)
  print *, 'Result at (1,1):', c_h(1,1)
  
  ! Clean up
  deallocate(a_h, b_h, c_h)
  call gpu_free(a)
  call gpu_free(b)
  call gpu_free(c)
  call gpu_blas_destroy(handle)
  
end program example_2d
```

**Important Note about BLAS Function Arguments:**

When calling BLAS functions, always use the address of the first element of the array (e.g., `x%f(1)` for 1D arrays or `a%f(1,1)` for 2D arrays), otherwise you may encounter type errors:

```f90
! Correct:
call gpu_ddot(handle, n, x%f(1), 1, y%f(1), 1, result)

! Incorrect (may cause type error):
call gpu_ddot(handle, n, x, 1, y, 1, result)
```

**Technical Note:** The library wrappers use Fortran's `c_loc()` intrinsic to obtain the memory address of the array element. By passing `x%f(1)`, you're providing the first element as a scalar with the `target` attribute, which `c_loc()` then converts to the appropriate C pointer for the underlying BLAS routines.

### Using Different Library Versions

To use a specific library version, link your application against the desired library:

```bash
# CPU version
gfortran -o myapp myapp.f90 -lgpu_cpu -lopenblas

# GPU version (NVIDIA)
gfortran -o myapp myapp.f90 -lgpu_nvidia -L/usr/local/cuda/lib64 -lcudart -lcublas
```

### Runtime Library Selection

For testing and comparison, you can dynamically load different libraries at runtime:

```f90
! Load CPU library
call load_library("libgpu_cpu.so")
! ... run computations ...

! Load GPU library
call load_library("libgpu_nvidia.so")
! ... run same computations and compare ...
```

## Testing

The library includes comprehensive unit tests that compare CPU and GPU implementations to ensure correctness.

Run tests with:
```bash
make check
```

For verbose test output:
```bash
make check-verbose
```

## API Reference

For detailed API documentation, see the comments in:
- `include/simple_gpu.h`: C interface declarations
- `include/simple_gpu.F90`: Fortran module and type definitions

## Performance Notes

- For small problem sizes, CPU version may be faster due to GPU overhead
- GPU version shows significant speedup for larger matrices/vectors
- Use streams for asynchronous operations to overlap computation and data transfer
- Keep data on GPU between operations to minimize transfer overhead

## Troubleshooting

### CUDA not detected during configuration

If CUDA is installed but not detected:
```bash
./configure --with-cuda=/path/to/cuda
```

### BLAS library not found

Specify BLAS library explicitly:
```bash
./configure --with-blas="-lopenblas"
```

### Runtime errors with GPU version

1. Ensure NVIDIA drivers are properly installed
2. Check that CUDA libraries are in your library path:
   ```bash
   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
   ```
3. Verify GPU is accessible with `nvidia-smi`

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing style and conventions
- All tests pass before submitting
- New features include appropriate tests

## License

See [LICENSE](LICENSE) file for details.

## Acknowledgments

This library provides bindings to:
- BLAS (Basic Linear Algebra Subprograms)
- NVIDIA cuBLAS (CUDA Basic Linear Algebra Subroutines)
