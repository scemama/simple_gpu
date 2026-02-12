ACLOCAL_AMFLAGS = -I m4
AM_CPPFLAGS = -I$(srcdir)/include

EXTRA_DIST = README.md LICENSE

include_HEADERS = include/simple_gpu.h include/simple_gpu.F90

SUBDIRS = src tests

.PHONY: check-verbose
check-verbose:
	$(MAKE) check VERBOSE=1
