ACLOCAL_AMFLAGS = -I m4
AM_CPPFLAGS = -I$(srcdir)/include

EXTRA_DIST = README.md LICENSE pkgconfig/simple_gpu-cpu.pc.in pkgconfig/simple_gpu-nvidia.pc.in

include_HEADERS = include/simple_gpu.h include/simple_gpu.F90

# pkgconfig files
pkgconfigdir = $(libdir)/pkgconfig
pkgconfig_DATA = pkgconfig/simple_gpu-cpu.pc

if BUILD_NVIDIA
pkgconfig_DATA += pkgconfig/simple_gpu-nvidia.pc
endif

# ===== Source library section (from src/Makefile.am) =====
lib_LTLIBRARIES = libsimple_gpu-cpu.la

# CPU version library
libsimple_gpu_cpu_la_SOURCES = src/gpu_cpu.c include/simple_gpu.F90
libsimple_gpu_cpu_la_CFLAGS = -I$(top_srcdir)/include
libsimple_gpu_cpu_la_FCFLAGS =
libsimple_gpu_cpu_la_LIBADD = $(BLAS_LIBS)
libsimple_gpu_cpu_la_LDFLAGS = -version-info 1:0:0

# NVIDIA GPU version library (conditional)
if BUILD_NVIDIA
lib_LTLIBRARIES += libsimple_gpu-nvidia.la
libsimple_gpu_nvidia_la_SOURCES = src/gpu_nvidia.c include/simple_gpu.F90
libsimple_gpu_nvidia_la_CFLAGS = -I$(top_srcdir)/include $(CUDA_CFLAGS)
libsimple_gpu_nvidia_la_FCFLAGS =
libsimple_gpu_nvidia_la_LIBADD = $(CUDA_LIBS)
libsimple_gpu_nvidia_la_LDFLAGS = -version-info 1:0:0 $(CUDA_LDFLAGS)
endif

# ===== Tests section (from tests/Makefile.am) =====
# Basic CPU tests (always built)
TESTS = tests/test_cpu
check_PROGRAMS = tests/test_cpu

tests_test_cpu_SOURCES = tests/test_basic.F90
tests_test_cpu_FCFLAGS = -I$(top_builddir)
tests_test_cpu_LDADD = libsimple_gpu-cpu.la

# GPU comparison tests (only when NVIDIA is available)
if BUILD_NVIDIA
TESTS += tests/test_gpu
check_PROGRAMS += tests/test_gpu

tests_test_gpu_SOURCES = tests/test_basic.F90
tests_test_gpu_FCFLAGS = -I$(top_builddir)
tests_test_gpu_LDADD = libsimple_gpu-nvidia.la
tests_test_gpu_LDFLAGS = -ldl
endif

# Clean up Fortran module files
CLEANFILES = *.mod

.PHONY: check-verbose
check-verbose:
	$(MAKE) check VERBOSE=1
