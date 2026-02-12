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

### Basic Example

```f90
program example
  use gpu
  implicit none
  
  type(gpu_blas) :: handle
  type(gpu_double1) :: x, y
  double precision :: result
  integer :: n
  
  n = 1000
  
  ! Initialize BLAS handle
  call gpu_blas_create(handle)
  
  ! Allocate vectors
  call gpu_allocate(x, n)
  call gpu_allocate(y, n)
  
  ! Initialize data (simplified)
  ! ... fill x%f and y%f with data ...
  
  ! Upload to GPU
  call gpu_upload(x)
  call gpu_upload(y)
  
  ! Compute dot product on GPU
  call gpu_ddot(handle, n, x, 1, y, 1, result)
  
  ! Clean up
  call gpu_free(x)
  call gpu_free(y)
  call gpu_blas_destroy(handle)
  
end program example
```

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

## Authors

- Anthony Scemama (@scemama)

## Acknowledgments

This library provides bindings to:
- BLAS (Basic Linear Algebra Subprograms)
- NVIDIA cuBLAS (CUDA Basic Linear Algebra Subroutines)
