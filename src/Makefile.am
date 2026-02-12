lib_LTLIBRARIES = libgpu_cpu.la

# CPU version library
libgpu_cpu_la_SOURCES = gpu_cpu.c gpu_module.F90
libgpu_cpu_la_CFLAGS = -I$(top_srcdir)/include
libgpu_cpu_la_FCFLAGS = 
libgpu_cpu_la_LIBADD = $(BLAS_LIBS)
libgpu_cpu_la_LDFLAGS = -version-info 1:0:0

# NVIDIA GPU version library (conditional)
if BUILD_NVIDIA
lib_LTLIBRARIES += libgpu_nvidia.la
libgpu_nvidia_la_SOURCES = gpu_nvidia.c gpu_module.F90
libgpu_nvidia_la_CFLAGS = -I$(top_srcdir)/include $(CUDA_CFLAGS)
libgpu_nvidia_la_FCFLAGS = 
libgpu_nvidia_la_LIBADD = $(CUDA_LIBS)
libgpu_nvidia_la_LDFLAGS = -version-info 1:0:0 $(CUDA_LDFLAGS)
endif

# Install header file
include_HEADERS = $(top_srcdir)/include/gpu.h

# Clean up Fortran module files
CLEANFILES = *.mod
