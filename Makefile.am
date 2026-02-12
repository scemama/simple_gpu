ACLOCAL_AMFLAGS = -I m4
AM_CPPFLAGS = -I$(srcdir)/include

EXTRA_DIST = README.md LICENSE

include_HEADERS = include/simple_gpu.h include/simple_gpu.F90

# ===== Source library section (from src/Makefile.am) =====
lib_LTLIBRARIES = libgpu_cpu.la

# CPU version library
libgpu_cpu_la_SOURCES = src/gpu_cpu.c include/simple_gpu.F90
libgpu_cpu_la_CFLAGS = -I$(top_srcdir)/include
libgpu_cpu_la_FCFLAGS =
libgpu_cpu_la_LIBADD = $(BLAS_LIBS)
libgpu_cpu_la_LDFLAGS = -version-info 1:0:0

# NVIDIA GPU version library (conditional)
if BUILD_NVIDIA
lib_LTLIBRARIES += libgpu_nvidia.la
libgpu_nvidia_la_SOURCES = src/gpu_nvidia.c include/simple_gpu.F90
libgpu_nvidia_la_CFLAGS = -I$(top_srcdir)/include $(CUDA_CFLAGS)
libgpu_nvidia_la_FCFLAGS =
libgpu_nvidia_la_LIBADD = $(CUDA_LIBS)
libgpu_nvidia_la_LDFLAGS = -version-info 1:0:0 $(CUDA_LDFLAGS)
endif

# ===== Tests section (from tests/Makefile.am) =====
# Basic CPU tests (always built)
TESTS = tests/test_cpu_basic
check_PROGRAMS = tests/test_cpu_basic

tests_test_cpu_basic_SOURCES = tests/test_cpu_basic.F90
tests_test_cpu_basic_FCFLAGS = -I$(top_builddir)
tests_test_cpu_basic_LDADD = libgpu_cpu.la

# GPU comparison tests (only when NVIDIA is available)
if BUILD_NVIDIA
TESTS += tests/test_gpu_comparison
check_PROGRAMS += tests/test_gpu_comparison

tests_test_gpu_comparison_SOURCES = tests/test_gpu_comparison.F90
tests_test_gpu_comparison_FCFLAGS = -I$(top_builddir)
tests_test_gpu_comparison_LDADD = libgpu_cpu.la
tests_test_gpu_comparison_LDFLAGS = -ldl
endif

# Clean up Fortran module files
CLEANFILES = *.mod

.PHONY: check-verbose
check-verbose:
	$(MAKE) check VERBOSE=1
