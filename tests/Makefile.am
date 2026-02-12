# Basic CPU tests (always built)
TESTS = test_cpu_basic
check_PROGRAMS = test_cpu_basic

test_cpu_basic_SOURCES = test_cpu_basic.F90
test_cpu_basic_FCFLAGS = -I$(top_builddir)/src
test_cpu_basic_LDADD = $(top_builddir)/src/libgpu_cpu.la

# GPU comparison tests (only when NVIDIA is available)
if BUILD_NVIDIA
TESTS += test_gpu_comparison
check_PROGRAMS += test_gpu_comparison

test_gpu_comparison_SOURCES = test_gpu_comparison.F90
test_gpu_comparison_FCFLAGS = -I$(top_builddir)/src
test_gpu_comparison_LDADD = $(top_builddir)/src/libgpu_cpu.la
test_gpu_comparison_LDFLAGS = -ldl
endif

CLEANFILES = *.mod
