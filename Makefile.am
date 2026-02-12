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
libsimple_gpu_nvidia_la_CFLAGS = -I$(top_srcdir)/include $(CUDA_CFLAGS) $(NVIDIA_CFLAGS)
libsimple_gpu_nvidia_la_FCFLAGS = $(NVIDIA_FCFLAGS)
libsimple_gpu_nvidia_la_LIBADD = $(CUDA_LIBS)
libsimple_gpu_nvidia_la_LDFLAGS = -version-info 1:0:0 $(CUDA_LDFLAGS)
endif

# ===== Tests section (from tests/Makefile.am) =====
# CPU tests (always built)
TESTS = \
	tests/test_ddot_cpu \
	tests/test_sdot_cpu \
	tests/test_dgemv_cpu \
	tests/test_sgemv_cpu \
	tests/test_dgemm_cpu \
	tests/test_sgemm_cpu \
	tests/test_dgeam_cpu \
	tests/test_sgeam_cpu

check_PROGRAMS = $(TESTS)

# DDOT tests
tests_test_ddot_cpu_SOURCES = tests/test_ddot.F90
tests_test_ddot_cpu_FCFLAGS = -I$(top_builddir)
tests_test_ddot_cpu_LDADD = libsimple_gpu-cpu.la

# SDOT tests
tests_test_sdot_cpu_SOURCES = tests/test_sdot.F90
tests_test_sdot_cpu_FCFLAGS = -I$(top_builddir)
tests_test_sdot_cpu_LDADD = libsimple_gpu-cpu.la

# DGEMV tests
tests_test_dgemv_cpu_SOURCES = tests/test_dgemv.F90
tests_test_dgemv_cpu_FCFLAGS = -I$(top_builddir)
tests_test_dgemv_cpu_LDADD = libsimple_gpu-cpu.la

# SGEMV tests
tests_test_sgemv_cpu_SOURCES = tests/test_sgemv.F90
tests_test_sgemv_cpu_FCFLAGS = -I$(top_builddir)
tests_test_sgemv_cpu_LDADD = libsimple_gpu-cpu.la

# DGEMM tests
tests_test_dgemm_cpu_SOURCES = tests/test_dgemm.F90
tests_test_dgemm_cpu_FCFLAGS = -I$(top_builddir)
tests_test_dgemm_cpu_LDADD = libsimple_gpu-cpu.la

# SGEMM tests
tests_test_sgemm_cpu_SOURCES = tests/test_sgemm.F90
tests_test_sgemm_cpu_FCFLAGS = -I$(top_builddir)
tests_test_sgemm_cpu_LDADD = libsimple_gpu-cpu.la

# DGEAM tests
tests_test_dgeam_cpu_SOURCES = tests/test_dgeam.F90
tests_test_dgeam_cpu_FCFLAGS = -I$(top_builddir)
tests_test_dgeam_cpu_LDADD = libsimple_gpu-cpu.la

# SGEAM tests
tests_test_sgeam_cpu_SOURCES = tests/test_sgeam.F90
tests_test_sgeam_cpu_FCFLAGS = -I$(top_builddir)
tests_test_sgeam_cpu_LDADD = libsimple_gpu-cpu.la

# GPU tests (only when NVIDIA is available)
if BUILD_NVIDIA
TESTS += \
	tests/test_ddot_gpu \
	tests/test_sdot_gpu \
	tests/test_dgemv_gpu \
	tests/test_sgemv_gpu \
	tests/test_dgemm_gpu \
	tests/test_sgemm_gpu \
	tests/test_dgeam_gpu \
	tests/test_sgeam_gpu

check_PROGRAMS += \
	tests/test_ddot_gpu \
	tests/test_sdot_gpu \
	tests/test_dgemv_gpu \
	tests/test_sgemv_gpu \
	tests/test_dgemm_gpu \
	tests/test_sgemm_gpu \
	tests/test_dgeam_gpu \
	tests/test_sgeam_gpu

# DDOT GPU test
tests_test_ddot_gpu_SOURCES = tests/test_ddot.F90
tests_test_ddot_gpu_FCFLAGS = -I$(top_builddir)
tests_test_ddot_gpu_LDADD = libsimple_gpu-nvidia.la
tests_test_ddot_gpu_LDFLAGS = -ldl

# SDOT GPU test
tests_test_sdot_gpu_SOURCES = tests/test_sdot.F90
tests_test_sdot_gpu_FCFLAGS = -I$(top_builddir)
tests_test_sdot_gpu_LDADD = libsimple_gpu-nvidia.la
tests_test_sdot_gpu_LDFLAGS = -ldl

# DGEMV GPU test
tests_test_dgemv_gpu_SOURCES = tests/test_dgemv.F90
tests_test_dgemv_gpu_FCFLAGS = -I$(top_builddir)
tests_test_dgemv_gpu_LDADD = libsimple_gpu-nvidia.la
tests_test_dgemv_gpu_LDFLAGS = -ldl

# SGEMV GPU test
tests_test_sgemv_gpu_SOURCES = tests/test_sgemv.F90
tests_test_sgemv_gpu_FCFLAGS = -I$(top_builddir)
tests_test_sgemv_gpu_LDADD = libsimple_gpu-nvidia.la
tests_test_sgemv_gpu_LDFLAGS = -ldl

# DGEMM GPU test
tests_test_dgemm_gpu_SOURCES = tests/test_dgemm.F90
tests_test_dgemm_gpu_FCFLAGS = -I$(top_builddir)
tests_test_dgemm_gpu_LDADD = libsimple_gpu-nvidia.la
tests_test_dgemm_gpu_LDFLAGS = -ldl

# SGEMM GPU test
tests_test_sgemm_gpu_SOURCES = tests/test_sgemm.F90
tests_test_sgemm_gpu_FCFLAGS = -I$(top_builddir)
tests_test_sgemm_gpu_LDADD = libsimple_gpu-nvidia.la
tests_test_sgemm_gpu_LDFLAGS = -ldl

# DGEAM GPU test
tests_test_dgeam_gpu_SOURCES = tests/test_dgeam.F90
tests_test_dgeam_gpu_FCFLAGS = -I$(top_builddir)
tests_test_dgeam_gpu_LDADD = libsimple_gpu-nvidia.la
tests_test_dgeam_gpu_LDFLAGS = -ldl

# SGEAM GPU test
tests_test_sgeam_gpu_SOURCES = tests/test_sgeam.F90
tests_test_sgeam_gpu_FCFLAGS = -I$(top_builddir)
tests_test_sgeam_gpu_LDADD = libsimple_gpu-nvidia.la
tests_test_sgeam_gpu_LDFLAGS = -ldl
endif

# Clean up Fortran module files
CLEANFILES = *.mod

.PHONY: check-verbose
check-verbose:
	$(MAKE) check VERBOSE=1
