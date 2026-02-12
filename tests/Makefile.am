TESTS = 
check_PROGRAMS = 

# Test programs will be added as they are created
if BUILD_NVIDIA
TESTS += test_gpu_comparison
check_PROGRAMS += test_gpu_comparison

test_gpu_comparison_SOURCES = test_gpu_comparison.F90
test_gpu_comparison_FCFLAGS = -I$(top_builddir)/src
test_gpu_comparison_LDADD = $(top_builddir)/src/libgpu_cpu.la
test_gpu_comparison_LDFLAGS = -ldl

CLEANFILES = *.mod
endif
